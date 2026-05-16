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
    : sc_module(name), cmd_in("cmd_in"), rsp_out("rsp_out"), m_memory(0) {
    SC_THREAD(command_thread);
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

void HDC_Accelerator::command_thread() {
    while (true) {
        const AccelCommand command = cmd_in.read();
        switch (command.kind) {
        case AccelCommandKind::ResetTraining:
            reset_training_state();
            break;

        case AccelCommandKind::ResetInference:
            reset_inference_state();
            break;

        case AccelCommandKind::TrainSample:
            push_training_sample(static_cast<int>(command.class_id.to_uint()), command.sample.levels);
            break;

        case AccelCommandKind::InvalidTrainingStep:
            push_invalid_training_step();
            break;

        case AccelCommandKind::InferSample: {
            distance_counter_t distances[NUM_CLASSES];
            const bool valid_prediction = push_inference_sample(command.sample.levels, distances);
            AccelResponse response;
            fill_inference_response(valid_prediction, distances, response);
            rsp_out.write(response);
            break;
        }

        case AccelCommandKind::Shutdown:
            return;
        }
    }
}

void HDC_Accelerator::reset_training_state() {
    reset_ngram_buffer();
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

void HDC_Accelerator::push_invalid_training_step() {
    finalize_current_class();
    reset_ngram_buffer();
}

void HDC_Accelerator::reset_inference_state() {
    reset_ngram_buffer();
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

void HDC_Accelerator::push_sample_to_ngram_buffer(const level_t *quantized_sample) {
    if (quantized_sample == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "quantized_sample must not be null");
    }

    encode_sample(quantized_sample, m_ngram_buffer[m_ngram_buffer_write_pos]);

    m_ngram_buffer_write_pos = (m_ngram_buffer_write_pos + 1) % N_GRAM_SIZE;
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        ++m_ngram_buffer_fill_count;
    }
}

void HDC_Accelerator::push_training_sample(int class_id, const level_t *quantized_sample) {
    if (class_id < 0 || class_id >= NUM_CLASSES) {
        SC_REPORT_FATAL("HDC_Accelerator", "class index out of range");
    }

    if (m_current_class_id < 0) {
        m_current_class_id = class_id;
    } else if (m_current_class_id != class_id) {
        SC_REPORT_FATAL("HDC_Accelerator", "class changed without invalid training step");
    }

    push_sample_to_ngram_buffer(quantized_sample);

    if (m_ngram_buffer_fill_count == N_GRAM_SIZE) {
        hv_t encoded_ngram;
        bind_ngram(encoded_ngram);
        add_ngram_to_bundling_buffer(encoded_ngram);
    }
}

bool HDC_Accelerator::push_inference_sample(const level_t *quantized_sample, distance_counter_t *distances) {
    if (distances == 0) {
        SC_REPORT_FATAL("HDC_Accelerator", "distances must not be null");
    }

    push_sample_to_ngram_buffer(quantized_sample);
    if (m_ngram_buffer_fill_count < N_GRAM_SIZE) {
        return false;
    }

    hv_t encoded_ngram;
    bind_ngram(encoded_ngram);
    compute_hamming_distances(encoded_ngram, distances);
    return true;
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
