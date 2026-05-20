#include "hdc_accelerator.h"

namespace hdc_systemc {

namespace {

bool get_bit(const hv_t &hv, int index) {
    return hv[index].to_bool();
}

void set_bit(hv_t &hv, int index, bool value) {
    hv[index] = value ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
}

void clear_hv(hv_t &hv) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

} // namespace

HDC_Accelerator::HDC_Accelerator(sc_core::sc_module_name name)
    : sc_module(name),
      cmd_in("cmd_in"),
      rsp_out("rsp_out"),
      m_encoder_in_fifo("encoder_in_fifo", 32),
      m_encoder_out_fifo("encoder_out_fifo", 32),
      m_bundler_in_fifo("bundler_in_fifo", 32),
      m_distance_in_fifo("distance_in_fifo", 32),
      m_control_done_fifo("control_done_fifo", 8),
      m_distance_done_fifo("distance_done_fifo", 32),
      m_infer_outstanding(0),
      m_memory(0) {
    reset_stats();

    SC_THREAD(command_thread);
    SC_THREAD(encoder_thread);
    SC_THREAD(ngram_thread);
    SC_THREAD(bundler_thread);
    SC_THREAD(distance_thread);

    reset_all_local_state();
}

void HDC_Accelerator::bind_memory(HDC_Memory *memory) {
    m_memory = memory;
}

void HDC_Accelerator::reset_stats() {
    m_stats.command_count = 0;
    m_stats.train_samples = 0;
    m_stats.infer_samples = 0;
    m_stats.encoded_samples = 0;
    m_stats.ngram_samples = 0;
    m_stats.valid_ngrams = 0;
    m_stats.bundled_ngrams = 0;
    m_stats.bundle_flushes = 0;
    m_stats.distance_requests = 0;
    m_stats.valid_distance_requests = 0;
}

const AcceleratorStats &HDC_Accelerator::stats() const {
    return m_stats;
}

// Data commands are pipelined: TrainSample and InferSample are dispatched
// without waiting for completion. Control commands are blocking stream
// boundaries and wait until their token passes through the internal pipeline.
void HDC_Accelerator::command_thread() {
    while (true) {
        forward_completed_distance_responses();

        AccelCommand command = {};
        if (!cmd_in.nb_read(command)) {
            if (m_distance_done_fifo.num_available() == 0) {
                sc_core::wait(cmd_in.data_written_event() | m_distance_done_fifo.data_written_event());
            }
            continue;
        }

        ++m_stats.command_count;
        switch (command.kind) {
        case AccelCommandKind::ResetTraining: {
            reset_bundling_buffer_only();
            PipelineItem item = {};
            item.kind = AccelCommandKind::ResetTraining;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            m_control_done_fifo.read();
            break;
        }

        case AccelCommandKind::ResetInference: {
            PipelineItem item = {};
            item.kind = AccelCommandKind::ResetInference;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            m_control_done_fifo.read();
            break;
        }

        case AccelCommandKind::TrainSample: {
            ++m_stats.train_samples;
            PipelineItem item = {};
            item.kind = AccelCommandKind::TrainSample;
            item.class_id = command.class_id;
            item.sample = command.sample;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            break;
        }

        case AccelCommandKind::InvalidTrainingStep: {
            // InvalidTrainingStep is a flush token for the current training class segment.
            // Since it uses the same FIFO path as samples, previous samples are bundled first.
            PipelineItem item = {};
            item.kind = AccelCommandKind::InvalidTrainingStep;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            m_control_done_fifo.read();
            break;
        }

        case AccelCommandKind::InferSample: {
            ++m_stats.infer_samples;
            PipelineItem item = {};
            item.kind = AccelCommandKind::InferSample;
            item.class_id = 0;
            item.sample = command.sample;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            ++m_infer_outstanding;
            break;
        }

        case AccelCommandKind::Shutdown: {
            while (m_infer_outstanding > 0) {
                forward_completed_distance_responses();
                if (m_infer_outstanding > 0) {
                    sc_core::wait(m_distance_done_fifo.data_written_event());
                }
            }

            PipelineItem shutdown = {};
            shutdown.kind = AccelCommandKind::Shutdown;
            shutdown.valid_ngram = false;
            m_encoder_in_fifo.write(shutdown);
            m_control_done_fifo.read();

            AccelResponse response = {};
            response.valid_prediction = false;
            response.is_shutdown_ack = true;
            response.predicted_class = 0;
            rsp_out.write(response);
            return;
        }
        }
    }
}

void HDC_Accelerator::forward_completed_distance_responses() {
    DistanceResponse distance_response = {};
    while (m_distance_done_fifo.nb_read(distance_response)) {
        AccelResponse response = {};
        response.valid_prediction = distance_response.valid_prediction;
        response.is_shutdown_ack = false;
        response.predicted_class = 0;
        for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
            response.distances[class_id] = distance_response.distances[class_id];
        }

        rsp_out.write(response);
        --m_infer_outstanding;
    }
}

void HDC_Accelerator::encoder_thread() {
    while (true) {
        PipelineItem item = m_encoder_in_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            m_encoder_out_fifo.write(item);
            return;
        }

        if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            encode_sample(item.sample, item.encoded);
            ++m_stats.encoded_samples;
        }
        m_encoder_out_fifo.write(item);
    }
}

void HDC_Accelerator::ngram_thread() {
    while (true) {
        PipelineItem item = m_encoder_out_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            m_distance_in_fifo.write(item);
            m_bundler_in_fifo.write(item);
            return;
        }

        if (item.kind == AccelCommandKind::ResetTraining) {
            reset_ngram_buffer();
            m_control_done_fifo.write(true);
            continue;
        }

        if (item.kind == AccelCommandKind::ResetInference) {
            reset_ngram_buffer();
            m_control_done_fifo.write(true);
            continue;
        }

        if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            reset_ngram_buffer();
            item.valid_ngram = false;
            m_bundler_in_fifo.write(item);
            continue;
        }

        if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            ++m_stats.ngram_samples;
            push_encoded_sample_to_ngram_buffer(item.encoded);
            if (m_ngram_buffer_fill_count == N_GRAM_SIZE) {
                bind_ngram(item.ngram);
                item.valid_ngram = true;
                ++m_stats.valid_ngrams;
            } else {
                item.valid_ngram = false;
            }

            if (item.kind == AccelCommandKind::TrainSample) {
                m_bundler_in_fifo.write(item);
            } else {
                m_distance_in_fifo.write(item);
            }
        }
    }
}

void HDC_Accelerator::bundler_thread() {
    while (true) {
        const PipelineItem item = m_bundler_in_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            m_control_done_fifo.write(true);
            return;
        }

        if (item.kind == AccelCommandKind::TrainSample) {
            if (item.valid_ngram) {
                const int class_id = item.class_id.to_int();

                if (m_current_class_id < 0) {
                    m_current_class_id = class_id;
                }

                add_ngram_to_bundling_buffer(item.ngram);
            }
            continue;
        }

        if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            finalize_current_class();
            reset_bundling_buffer_only();
            ++m_stats.bundle_flushes;
            m_control_done_fifo.write(true);
            continue;
        }
    }
}

void HDC_Accelerator::distance_thread() {
    while (true) {
        const PipelineItem item = m_distance_in_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            return;
        }

        DistanceResponse response = {};
        ++m_stats.distance_requests;
        if (!item.valid_ngram) {
            response.valid_prediction = false;
            for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                response.distances[class_id] = 0;
            }
            m_distance_done_fifo.write(response);
            continue;
        }

        response.valid_prediction = true;
        ++m_stats.valid_distance_requests;
        compute_hamming_distances(item.ngram, response.distances);
        m_distance_done_fifo.write(response);
    }
}

void HDC_Accelerator::reset_all_local_state() {
    reset_ngram_buffer();
    reset_bundling_buffer_only();
}

void HDC_Accelerator::reset_bundling_buffer_only() {
    m_current_class_count = 0;
    m_current_class_id = -1;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        m_bundling_score[d] = 0;
    }
}

void HDC_Accelerator::reset_ngram_buffer() {
    m_ngram_buffer_write_pos = 0;
    m_ngram_buffer_fill_count = 0;
    for (int slot = 0; slot < N_GRAM_SIZE; ++slot) {
        clear_hv(m_ngram_buffer[slot]);
    }
}

void HDC_Accelerator::add_ngram_to_bundling_buffer(const hv_t &encoded_ngram) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        if (get_bit(encoded_ngram, d)) {
            ++m_bundling_score[d];
        } else {
            --m_bundling_score[d];
        }
    }
    ++m_current_class_count;
    ++m_stats.bundled_ngrams;
}

void HDC_Accelerator::finalize_current_class() {
    hv_t class_vector;
    clear_hv(class_vector);
    // Exact equivalent of the previous rule:
    //     ones >= floor(half of m_current_class_count)
    //
    // Signed score:
    //     score = ones - zeros = 2 * ones - m_current_class_count
    //
    // Therefore:
    //     even count: score >= 0
    //     odd count:  score >= -1
    //
    // This avoids division while preserving the old bundling result.
    const bool odd_count = (m_current_class_count.to_uint() & 1u) != 0u;
    const train_score_t signed_threshold = odd_count ? train_score_t(-1) : train_score_t(0);
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(class_vector, d, m_bundling_score[d] >= signed_threshold);
        m_bundling_score[d] = 0;
    }
    m_memory->write_assoc_class(static_cast<unsigned>(m_current_class_id), class_vector);

    m_current_class_count = 0;
    m_current_class_id = -1;
}

void HDC_Accelerator::bind_ngram(hv_t &encoded_ngram) {
    const int oldest_slot = m_ngram_buffer_write_pos;
    encoded_ngram = m_ngram_buffer[oldest_slot];
    hv_t next_encoded;

    for (int i = 1; i < N_GRAM_SIZE; ++i) {
        const int slot = (oldest_slot + i) % N_GRAM_SIZE;
        permute_xor(encoded_ngram, m_ngram_buffer[slot], next_encoded);
        encoded_ngram = next_encoded;
    }
}

void HDC_Accelerator::permute_xor(const hv_t &input, const hv_t &rhs, hv_t &output) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        const int source_index = (d + VECTOR_DIMENSION - 1) % VECTOR_DIMENSION;
        const bool bit = get_bit(input, source_index) ^ get_bit(rhs, d);
        set_bit(output, d, bit);
    }
}

void HDC_Accelerator::push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample) {
    m_ngram_buffer[m_ngram_buffer_write_pos] = encoded_sample;
    m_ngram_buffer_write_pos = (m_ngram_buffer_write_pos + 1) % N_GRAM_SIZE;
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        ++m_ngram_buffer_fill_count;
    }
}

void HDC_Accelerator::encode_sample(const QuantizedSample &sample, hv_t &encoded_sample) {
    // Feature bundling uses signed +1/-1 accumulation instead of counting ones.
    // This is exactly equivalent to:
    //     ones >= floor(half of NUM_FEATURES)
    //
    // For NUM_FEATURES even: score >= 0
    // For NUM_FEATURES odd:  score >= -1
    const feature_score_t signed_threshold =
        (NUM_FEATURES % 2 == 1) ? feature_score_t(-1) : feature_score_t(0);

    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        feature_score_t score = 0;
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &feature_hv = m_memory->read_cim(sample.levels[feature], feature);
            if (get_bit(feature_hv, d)) {
                ++score;
            } else {
                --score;
            }
        }

        set_bit(encoded_sample, d, score >= signed_threshold);
    }
}

void HDC_Accelerator::compute_hamming_distances(const hv_t &query, distance_counter_t *distances) {
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const hv_t &class_vector = m_memory->read_assoc_class(static_cast<unsigned>(class_id));
        distance_counter_t distance = 0;
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vector, d)) {
                ++distance;
            }
        }
        distances[class_id] = distance;
    }
}

} // namespace hdc_systemc
