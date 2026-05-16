#include "hdc_accelerator.h"

namespace hdc_systemc {

namespace {

bool get_bit(const hv_t &hv, int index) {
    return hv[index].to_bool();
}

void set_bit(hv_t &hv, int index, bool value) {
    hv[index] = value ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
}

void xor_hv(const hv_t &lhs, const hv_t &rhs, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(dst, d, get_bit(lhs, d) ^ get_bit(rhs, d));
    }
}

void clear_hv(hv_t &hv) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

void permute_right(const hv_t &src, unsigned shift, hv_t &dst) {
    const unsigned effective_shift = shift % VECTOR_DIMENSION;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        const int out_index = (d + static_cast<int>(effective_shift)) % VECTOR_DIMENSION;
        dst[out_index] = src[d];
    }
}

} // namespace

HDC_Accelerator::HDC_Accelerator(sc_core::sc_module_name name)
    : sc_module(name),
      cmd_in("cmd_in"),
      rsp_out("rsp_out"),
      m_encoder_in_fifo("encoder_in_fifo", 8),
      m_encoder_out_fifo("encoder_out_fifo", 8),
      m_bundler_in_fifo("bundler_in_fifo", 8),
      m_distance_in_fifo("distance_in_fifo", 8),
      m_control_done_fifo("control_done_fifo", 8),
      m_distance_done_fifo("distance_done_fifo", 8),
      m_memory(0) {
    SC_THREAD(command_thread);
    SC_THREAD(encoder_thread);
    SC_THREAD(ngram_thread);
    SC_THREAD(bundler_thread);
    SC_THREAD(distance_thread);
    reset_training_state();
}

void HDC_Accelerator::bind_memory(HDC_Memory *memory) {
    m_memory = memory;
}

void HDC_Accelerator::fill_inference_response(bool valid_prediction,
                                              const distance_counter_t *distances,
                                              AccelResponse &response) const {
    response.valid_prediction = valid_prediction;
    response.predicted_class = 0;

    distance_counter_t best_distance = 0;
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const distance_counter_t distance =
            (valid_prediction && distances != 0) ? distances[class_id] : distance_counter_t(0);
        response.distances[class_id] = distance;
        if (valid_prediction && (class_id == 0 || distance < best_distance)) {
            best_distance = distance;
            response.predicted_class = static_cast<unsigned>(class_id);
        }
    }
}

void HDC_Accelerator::fill_distance_response(bool valid_prediction,
                                             const distance_counter_t *distances,
                                             DistanceResponse &response) const {
    response.valid_prediction = valid_prediction;
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        response.distances[class_id] =
            (valid_prediction && distances != 0) ? distances[class_id] : distance_counter_t(0);
    }
}

void HDC_Accelerator::command_thread() {
    while (true) {
        const AccelCommand command = cmd_in.read();
        switch (command.kind) {
        case AccelCommandKind::ResetTraining: {
            reset_bundling_state();
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
            PipelineItem item = {};
            item.kind = AccelCommandKind::TrainSample;
            item.class_id = command.class_id;
            item.sample = command.sample;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            m_control_done_fifo.read();
            break;
        }

        case AccelCommandKind::InvalidTrainingStep: {
            PipelineItem item = {};
            item.kind = AccelCommandKind::InvalidTrainingStep;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);
            m_control_done_fifo.read();
            break;
        }

        case AccelCommandKind::InferSample: {
            PipelineItem item = {};
            item.kind = AccelCommandKind::InferSample;
            item.class_id = 0;
            item.sample = command.sample;
            item.valid_ngram = false;
            m_encoder_in_fifo.write(item);

            const DistanceResponse distance_response = m_distance_done_fifo.read();
            AccelResponse response = {};
            response.valid_prediction = distance_response.valid_prediction;
            response.predicted_class = 0;
            for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                response.distances[class_id] = distance_response.distances[class_id];
            }
            rsp_out.write(response);
            break;
        }

        case AccelCommandKind::Shutdown:
            PipelineItem shutdown = {};
            shutdown.kind = AccelCommandKind::Shutdown;
            m_encoder_in_fifo.write(shutdown);
            return;
        }
    }
}

void HDC_Accelerator::encoder_thread() {
    while (true) {
        PipelineItem item = m_encoder_in_fifo.read();
        if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            encode_sample(item.sample.levels, item.encoded);
            sc_core::wait(ACCEL_LATENCY_ENCODE_NS, sc_core::SC_NS);
        }
        m_encoder_out_fifo.write(item);
    }
}

void HDC_Accelerator::ngram_thread() {
    while (true) {
        PipelineItem item = m_encoder_out_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            m_bundler_in_fifo.write(item);
            m_distance_in_fifo.write(item);
            return;
        }

        if (item.kind == AccelCommandKind::ResetTraining || item.kind == AccelCommandKind::ResetInference) {
            reset_ngram_buffer();
            item.valid_ngram = false;
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
            push_encoded_sample_to_ngram_buffer(item.encoded);
            if (m_ngram_buffer_fill_count == N_GRAM_SIZE) {
                bind_ngram(item.ngram);
                item.valid_ngram = true;
            } else {
                item.valid_ngram = false;
            }

            sc_core::wait(ACCEL_LATENCY_NGRAM_NS, sc_core::SC_NS);

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
            return;
        }

        switch (item.kind) {
        case AccelCommandKind::TrainSample: {
            const int class_id = static_cast<int>(item.class_id.to_uint());
            if (class_id < 0 || class_id >= NUM_CLASSES) {
                SC_REPORT_FATAL("HDC_Accelerator", "class index out of range");
            }
            if (m_current_class_id < 0) {
                m_current_class_id = class_id;
            } else if (m_current_class_id != class_id) {
                SC_REPORT_FATAL("HDC_Accelerator", "class changed without invalid training step");
            }
            if (item.valid_ngram) {
                add_ngram_to_bundling_buffer(item.ngram);
            }
            m_control_done_fifo.write(true);
            break;
        }

        case AccelCommandKind::InvalidTrainingStep:
            finalize_current_class();
            m_control_done_fifo.write(true);
            break;

        case AccelCommandKind::ResetInference:
        case AccelCommandKind::ResetTraining:
        case AccelCommandKind::InferSample:
        case AccelCommandKind::Shutdown:
            break;
        }
    }
}

void HDC_Accelerator::distance_thread() {
    while (true) {
        const PipelineItem item = m_distance_in_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            return;
        }

        if (item.kind == AccelCommandKind::InferSample) {
            distance_counter_t distances[NUM_CLASSES];
            if (item.valid_ngram) {
                compute_hamming_distances(item.ngram, distances);
                DistanceResponse response;
                fill_distance_response(true, distances, response);
                m_distance_done_fifo.write(response);
            } else {
                DistanceResponse response;
                fill_distance_response(false, 0, response);
                m_distance_done_fifo.write(response);
            }
        }
    }
}

void HDC_Accelerator::reset_training_state() {
    reset_ngram_buffer();
    reset_bundling_state();
}

void HDC_Accelerator::reset_bundling_state() {
    m_current_class_count = 0;
    m_current_class_id = -1;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        m_bundling_buffer[d] = 0;
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
            ++m_bundling_buffer[d];
        }
    }
    ++m_current_class_count;
}

void HDC_Accelerator::finalize_current_class() {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (m_current_class_id < 0) {
        return;
    }
    if (m_current_class_id >= NUM_CLASSES) {
        SC_REPORT_FATAL("HDC_Accelerator", "active class index out of range");
    }
    if (m_current_class_count == 0) {
        m_current_class_id = -1;
        return;
    }

    hv_t class_vector;
    clear_hv(class_vector);
    const train_counter_t threshold = m_current_class_count / 2;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(class_vector, d, m_bundling_buffer[d] >= threshold);
        m_bundling_buffer[d] = 0;
    }
    m_memory->write_assoc_class(static_cast<unsigned>(m_current_class_id), class_vector);

    m_current_class_count = 0;
    m_current_class_id = -1;
}

void HDC_Accelerator::bind_ngram(hv_t &encoded) const {
    const int oldest_slot = m_ngram_buffer_write_pos;
    encoded = m_ngram_buffer[oldest_slot];

    hv_t permuted_result;
    for (int i = 1; i < N_GRAM_SIZE; ++i) {
        const int slot = (oldest_slot + i) % N_GRAM_SIZE;
        permute_right(encoded, 1u, permuted_result);
        xor_hv(permuted_result, m_ngram_buffer[slot], encoded);
    }
}

void HDC_Accelerator::push_encoded_sample_to_ngram_buffer(const hv_t &encoded_sample) {
    m_ngram_buffer[m_ngram_buffer_write_pos] = encoded_sample;
    m_ngram_buffer_write_pos = (m_ngram_buffer_write_pos + 1) % N_GRAM_SIZE;
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        ++m_ngram_buffer_fill_count;
    }
}

void HDC_Accelerator::encode_sample(const level_t *quantized_sample, hv_t &encoded_sample) const {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (quantized_sample == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "quantized_sample must not be null");
    }

    const feature_counter_t threshold = NUM_FEATURES / 2;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        feature_counter_t ones = 0;
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &feature_hv = m_memory->read_cim(quantized_sample[feature], static_cast<unsigned>(feature));
            if (get_bit(feature_hv, d)) {
                ++ones;
            }
        }
        set_bit(encoded_sample, d, ones >= threshold);
    }
}

void HDC_Accelerator::compute_hamming_distances(const hv_t &query, distance_counter_t *distances) const {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (distances == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "distances must not be null");
    }

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        distance_counter_t distance = 0;
        const hv_t &class_vector = m_memory->read_assoc_class(static_cast<unsigned>(class_id));
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vector, d)) {
                ++distance;
            }
        }
        distances[class_id] = distance;
    }
}

} // namespace hdc_systemc
