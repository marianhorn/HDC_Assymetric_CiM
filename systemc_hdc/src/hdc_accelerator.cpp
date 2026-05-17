#include "hdc_accelerator.h"
#include "sysc/kernel/sc_spawn.h"

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
    for (unsigned pe = 0; pe < ENCODER_PES; ++pe) {
        m_encode_done_flags[pe] = false;
    }
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        m_distance_current_result[class_id] = 0;
        m_distance_done_flags[class_id] = false;
    }

    SC_THREAD(command_thread);
    SC_THREAD(encoder_thread);
    SC_THREAD(ngram_thread);
    SC_THREAD(bundler_thread);
    SC_THREAD(distance_thread);
    for (unsigned pe = 0; pe < ENCODER_PES; ++pe) {
        sc_core::sc_spawn(
            sc_core::sc_bind(&HDC_Accelerator::encoder_pe_thread, this, pe),
            sc_core::sc_gen_unique_name("encoder_pe"));
    }
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        sc_core::sc_spawn(
            sc_core::sc_bind(&HDC_Accelerator::distance_class_pe_thread, this, class_id),
            sc_core::sc_gen_unique_name("distance_class_pe"));
    }

    reset_training_state_local();
}

void HDC_Accelerator::bind_memory(HDC_Memory *memory) {
    m_memory = memory;
}

void HDC_Accelerator::command_thread() {
    while (true) {
        const AccelCommand command = cmd_in.read();
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
            response.is_shutdown_ack = false;
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

void HDC_Accelerator::encoder_thread() {
    while (true) {
        PipelineItem item = m_encoder_in_fifo.read();
        if (item.kind == AccelCommandKind::Shutdown) {
            m_encoder_out_fifo.write(item);
            return;
        }

        if (item.kind == AccelCommandKind::TrainSample || item.kind == AccelCommandKind::InferSample) {
            encode_sample_parallel(item.sample, item.encoded);
            sc_core::wait(ACCEL_LATENCY_ENCODE_NS, sc_core::SC_NS);
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
            m_control_done_fifo.write(true);
            return;
        }

        if (item.kind == AccelCommandKind::TrainSample) {
            if (item.valid_ngram) {
                const int class_id = item.class_id.to_int();
                if (class_id < 0 || class_id >= NUM_CLASSES) {
                    SC_REPORT_FATAL("HDC_Accelerator", "class index out of range");
                }

                if (m_current_class_id < 0) {
                    m_current_class_id = class_id;
                }
                if (class_id != m_current_class_id) {
                    SC_REPORT_FATAL("HDC_Accelerator", "Training class changed without flush");
                }

                add_ngram_to_bundling_buffer(item.ngram);
            }
            sc_core::wait(ACCEL_LATENCY_BUNDLE_NS, sc_core::SC_NS);
            m_control_done_fifo.write(true);
            continue;
        }

        if (item.kind == AccelCommandKind::InvalidTrainingStep) {
            finalize_current_class();
            reset_bundling_buffer_only();
            sc_core::wait(ACCEL_LATENCY_BUNDLE_NS, sc_core::SC_NS);
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

        DistanceResponse response;
        if (!item.valid_ngram) {
            response.valid_prediction = false;
            for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
                response.distances[class_id] = 0;
            }
            m_distance_done_fifo.write(response);
            continue;
        }

        response.valid_prediction = true;
        compute_hamming_distances_parallel(item.ngram, response.distances);
        sc_core::wait(ACCEL_LATENCY_DISTANCE_NS, sc_core::SC_NS);
        m_distance_done_fifo.write(response);
    }
}

void HDC_Accelerator::reset_training_state_local() {
    reset_ngram_buffer();
    reset_bundling_buffer_only();
}

void HDC_Accelerator::reset_bundling_buffer_only() {
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

void HDC_Accelerator::encode_sample_parallel(const QuantizedSample &sample, hv_t &encoded_sample) {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }

    m_encode_current_sample = sample;
    for (unsigned pe = 0; pe < ENCODER_PES; ++pe) {
        m_encode_done_flags[pe] = false;
    }

    m_encode_start_event.notify(sc_core::SC_ZERO_TIME);
    for (unsigned pe = 0; pe < ENCODER_PES; ++pe) {
        while (!m_encode_done_flags[pe]) {
            sc_core::wait(m_encode_done_event[pe]);
        }
    }

    encoded_sample = m_encode_current_output;
}

void HDC_Accelerator::encoder_pe_thread(unsigned pe_id) {
    while (true) {
        sc_core::wait(m_encode_start_event);

        const unsigned begin = pe_id * VECTOR_DIMENSION / ENCODER_PES;
        const unsigned end = (pe_id + 1) * VECTOR_DIMENSION / ENCODER_PES;
        const feature_counter_t threshold = NUM_FEATURES / 2;

        for (unsigned d = begin; d < end; ++d) {
            feature_counter_t ones = 0;
            for (unsigned feature = 0; feature < NUM_FEATURES; ++feature) {
                const hv_t &feature_hv =
                    m_memory->read_cim(m_encode_current_sample.levels[feature], feature);
                if (get_bit(feature_hv, static_cast<int>(d))) {
                    ++ones;
                }
            }

            set_bit(m_encode_current_output, static_cast<int>(d), ones >= threshold);
        }

        m_encode_done_flags[pe_id] = true;
        m_encode_done_event[pe_id].notify(sc_core::SC_ZERO_TIME);
    }
}

void HDC_Accelerator::compute_hamming_distances_parallel(const hv_t &query, distance_counter_t *distances) {
    if (m_memory == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "memory not bound");
    }
    if (distances == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "distances must not be null");
    }

    m_distance_current_query = query;
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        m_distance_done_flags[class_id] = false;
    }

    m_distance_start_event.notify(sc_core::SC_ZERO_TIME);
    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        while (!m_distance_done_flags[class_id]) {
            sc_core::wait(m_distance_done_event[class_id]);
        }
    }

    for (unsigned class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        distances[class_id] = m_distance_current_result[class_id];
    }
}

void HDC_Accelerator::distance_class_pe_thread(unsigned class_id) {
    while (true) {
        sc_core::wait(m_distance_start_event);

        const hv_t &class_vector = m_memory->read_assoc_class(class_id);
        distance_counter_t distance = 0;
        for (unsigned d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(m_distance_current_query, static_cast<int>(d)) !=
                get_bit(class_vector, static_cast<int>(d))) {
                ++distance;
            }
        }

        m_distance_current_result[class_id] = distance;
        m_distance_done_flags[class_id] = true;
        m_distance_done_event[class_id].notify(sc_core::SC_ZERO_TIME);
    }
}

} // namespace hdc_systemc
