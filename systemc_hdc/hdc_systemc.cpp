#include "hdc_systemc.h"

namespace hdc_systemc {

namespace {

void clear_hv(hv_t &hv) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

void copy_hv(const hv_t &src, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        dst[d] = src[d];
    }
}

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

void permute_right(const hv_t &src, unsigned shift, hv_t &dst) {
    const unsigned effective_shift = shift % VECTOR_DIMENSION;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        const int out_index = (d + static_cast<int>(effective_shift)) % VECTOR_DIMENSION;
        dst[out_index] = src[d];
    }
}

} // namespace

CiMMemory::CiMMemory(sc_core::sc_module_name name) : sc_module(name) {
    clear();
}

void CiMMemory::clear() {
    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        clear_hv(m_storage[i]);
    }
}

void CiMMemory::load(const hv_t *flat_cim) {
    if (flat_cim == 0) {
        SC_REPORT_FATAL("CiMMemory", "flat_cim must not be null");
    }

    for (int i = 0; i < NUM_LEVELS * NUM_FEATURES; ++i) {
        copy_hv(flat_cim[i], m_storage[i]);
    }
}

const hv_t &CiMMemory::read(level_t level, unsigned feature) const {
    const unsigned level_index = level.to_uint();
    if (level_index >= NUM_LEVELS) {
        SC_REPORT_FATAL("CiMMemory", "level index out of range");
        return m_storage[0];
    }
    if (feature >= static_cast<unsigned>(NUM_FEATURES)) {
        SC_REPORT_FATAL("CiMMemory", "feature index out of range");
        return m_storage[0];
    }

    return m_storage[(level_index * NUM_FEATURES) + static_cast<int>(feature)];
}

TimestampEncoder::TimestampEncoder(sc_core::sc_module_name name)
    : sc_module(name), m_cim(0) {}

void TimestampEncoder::bind_cim(const CiMMemory *cim) {
    m_cim = cim;
}

void TimestampEncoder::encode(const level_t *frame_levels, hv_t &encoded) const {
    if (m_cim == 0) {
        SC_REPORT_FATAL("TimestampEncoder", "CiM memory not bound");
    }
    if (frame_levels == 0) {
        SC_REPORT_FATAL("TimestampEncoder", "frame_levels must not be null");
    }

    const feature_counter_t threshold = NUM_FEATURES / 2;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        feature_counter_t ones = 0;
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &feature_hv = m_cim->read(frame_levels[feature], static_cast<unsigned>(feature));
            if (get_bit(feature_hv, d)) {
                ones = ones + 1;
            }
        }
        set_bit(encoded, d, ones >= threshold);
    }
}

FusionNGramEncoder::FusionNGramEncoder(sc_core::sc_module_name name)
    : sc_module(name), m_timestamp_encoder(0) {}

void FusionNGramEncoder::bind_timestamp_encoder(const TimestampEncoder *timestamp_encoder) {
    m_timestamp_encoder = timestamp_encoder;
}

void FusionNGramEncoder::encode_window(const level_t *level_window, hv_t &encoded) const {
    if (m_timestamp_encoder == 0) {
        SC_REPORT_FATAL("FusionNGramEncoder", "Timestamp encoder not bound");
    }
    if (level_window == 0) {
        SC_REPORT_FATAL("FusionNGramEncoder", "level_window must not be null");
    }

    m_timestamp_encoder->encode(level_window, encoded);

    hv_t encoded_timestamp;
    hv_t permuted_result;
    for (int i = 1; i < N_GRAM_SIZE; ++i) {
        m_timestamp_encoder->encode(&level_window[i * NUM_FEATURES], encoded_timestamp);
        permute_right(encoded, 1u, permuted_result);
        xor_hv(permuted_result, encoded_timestamp, encoded);
    }
}

AssocMemTrainer::AssocMemTrainer(sc_core::sc_module_name name) : sc_module(name) {
    reset();
}

void AssocMemTrainer::reset() {
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        m_vector_counts[class_id] = 0;
        clear_hv(m_class_vectors[class_id]);
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            m_class_bit_counts[class_id][d] = 0;
        }
    }
}

void AssocMemTrainer::accumulate(int class_id, const hv_t &encoded_ngram) {
    if (class_id < 0 || class_id >= NUM_CLASSES) {
        return;
    }

    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        if (get_bit(encoded_ngram, d)) {
            m_class_bit_counts[class_id][d] = m_class_bit_counts[class_id][d] + 1;
        }
    }
    m_vector_counts[class_id] = m_vector_counts[class_id] + 1;
}

void AssocMemTrainer::finalize() {
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const train_counter_t threshold = m_vector_counts[class_id] / 2;
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            set_bit(m_class_vectors[class_id], d, m_class_bit_counts[class_id][d] >= threshold);
        }
    }
}

const hv_t &AssocMemTrainer::get_class_vector(unsigned class_id) const {
    if (class_id >= static_cast<unsigned>(NUM_CLASSES)) {
        SC_REPORT_FATAL("AssocMemTrainer", "class index out of range");
        return m_class_vectors[0];
    }
    return m_class_vectors[class_id];
}

train_counter_t AssocMemTrainer::get_vector_count(unsigned class_id) const {
    if (class_id >= static_cast<unsigned>(NUM_CLASSES)) {
        SC_REPORT_FATAL("AssocMemTrainer", "class index out of range");
        return 0;
    }
    return m_vector_counts[class_id];
}

Classifier::Classifier(sc_core::sc_module_name name)
    : sc_module(name), m_assoc_mem(0) {}

void Classifier::bind_assoc_mem(const AssocMemTrainer *assoc_mem) {
    m_assoc_mem = assoc_mem;
}

void Classifier::compute_distances(const hv_t &query, distance_counter_t *distances) const {
    if (m_assoc_mem == 0) {
        SC_REPORT_FATAL("Classifier", "Associative memory not bound");
    }
    if (distances == 0) {
        SC_REPORT_FATAL("Classifier", "distances must not be null");
    }

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        distance_counter_t distance = 0;
        const hv_t &class_vector = m_assoc_mem->get_class_vector(static_cast<unsigned>(class_id));
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vector, d)) {
                distance = distance + 1;
            }
        }
        distances[class_id] = distance;
    }
}

int Classifier::predict(const hv_t &query) const {
    distance_counter_t distances[NUM_CLASSES];
    compute_distances(query, distances);

    int best_class = 0;
    distance_counter_t best_distance = distances[0];
    for (int class_id = 1; class_id < NUM_CLASSES; ++class_id) {
        if (distances[class_id] < best_distance) {
            best_distance = distances[class_id];
            best_class = class_id;
        }
    }
    return best_class;
}

HdcTop::HdcTop(sc_core::sc_module_name name)
    : sc_module(name),
      m_cim("cim_memory"),
      m_timestamp_encoder("timestamp_encoder"),
      m_ngram_encoder("fusion_ngram_encoder"),
      m_assoc_mem_trainer("assoc_mem_trainer"),
      m_classifier("classifier") {
    m_timestamp_encoder.bind_cim(&m_cim);
    m_ngram_encoder.bind_timestamp_encoder(&m_timestamp_encoder);
    m_classifier.bind_assoc_mem(&m_assoc_mem_trainer);
}

void HdcTop::load_cim(const hv_t *flat_cim) {
    m_cim.load(flat_cim);
}

void HdcTop::reset_assoc_mem() {
    m_assoc_mem_trainer.reset();
}

bool HdcTop::is_window_stable(const int *labels) const {
    if (labels == 0) {
        SC_REPORT_FATAL("HdcTop", "labels must not be null");
    }
    return labels[0] == labels[N_GRAM_SIZE - 1];
}

void HdcTop::train_dataset(const level_t *level_data, const int *labels, int num_samples) {
    if (level_data == 0 || labels == 0) {
        SC_REPORT_FATAL("HdcTop", "training data and labels must not be null");
    }

    hv_t encoded_ngram;
    for (int j = 0; j < num_samples - N_GRAM_SIZE; ++j) {
        if (is_window_stable(&labels[j])) {
            const int class_id = labels[j];
            encode_ngram(&level_data[j * NUM_FEATURES], encoded_ngram);
            m_assoc_mem_trainer.accumulate(class_id, encoded_ngram);
        } else {
            j += (N_GRAM_SIZE - 1);
        }
    }
}

void HdcTop::finalize_assoc_mem() {
    m_assoc_mem_trainer.finalize();
}

void HdcTop::encode_timestamp(const level_t *frame_levels, hv_t &encoded) const {
    m_timestamp_encoder.encode(frame_levels, encoded);
}

void HdcTop::encode_ngram(const level_t *level_window, hv_t &encoded) const {
    m_ngram_encoder.encode_window(level_window, encoded);
}

int HdcTop::predict_ngram(const level_t *level_window) const {
    hv_t encoded_ngram;
    encode_ngram(level_window, encoded_ngram);
    return m_classifier.predict(encoded_ngram);
}

void HdcTop::compute_distances_for_ngram(const level_t *level_window, distance_counter_t *distances) const {
    hv_t encoded_ngram;
    encode_ngram(level_window, encoded_ngram);
    m_classifier.compute_distances(encoded_ngram, distances);
}

const hv_t &HdcTop::get_class_vector(unsigned class_id) const {
    return m_assoc_mem_trainer.get_class_vector(class_id);
}

train_counter_t HdcTop::get_class_vector_count(unsigned class_id) const {
    return m_assoc_mem_trainer.get_vector_count(class_id);
}

int mode_smallest_tie(const int *labels, int size) {
    if (labels == 0 || size <= 0) {
        SC_REPORT_FATAL("mode_smallest_tie", "invalid labels input");
        return 0;
    }

    int max_value = 0;
    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        int count = 0;
        for (int j = 0; j < size; ++j) {
            if (labels[j] == labels[i]) {
                ++count;
            }
        }
        if (count > max_count) {
            max_count = count;
            max_value = labels[i];
        } else if (count == max_count && labels[i] < max_value) {
            max_value = labels[i];
        }
    }

    return max_value;
}

EvaluationResult evaluate_dataset(HdcTop &top, const level_t *level_data, const int *labels, int num_samples) {
    if (level_data == 0 || labels == 0) {
        SC_REPORT_FATAL("evaluate_dataset", "evaluation data and labels must not be null");
    }

    EvaluationResult result;
    result.correct = 0;
    result.not_correct = 0;
    result.transition_error = 0;
    result.total = 0;
    result.overall_accuracy = 0.0;
    result.non_transition_accuracy = 0.0;
    for (int i = 0; i < NUM_CLASSES; ++i) {
        for (int j = 0; j < NUM_CLASSES; ++j) {
            result.confusion_matrix[i][j] = 0;
        }
    }

    for (int j = 0; j < num_samples - N_GRAM_SIZE + 1; j += N_GRAM_SIZE) {
        const int actual_label = mode_smallest_tie(&labels[j], N_GRAM_SIZE);
        const int predicted_label = top.predict_ngram(&level_data[j * NUM_FEATURES]);

        if (actual_label >= 0 && actual_label < NUM_CLASSES &&
            predicted_label >= 0 && predicted_label < NUM_CLASSES) {
            result.confusion_matrix[actual_label][predicted_label]++;
        }

        if (predicted_label == actual_label) {
            result.correct++;
        } else if (labels[j] != labels[j + N_GRAM_SIZE - 1]) {
            result.transition_error++;
        } else {
            result.not_correct++;
        }
    }

    result.total = result.correct + result.not_correct + result.transition_error;
    if (result.total > 0) {
        result.overall_accuracy = static_cast<double>(result.correct) / static_cast<double>(result.total);
    }

    const unsigned non_transition_total = result.correct + result.not_correct;
    if (non_transition_total > 0) {
        result.non_transition_accuracy =
            static_cast<double>(result.correct) / static_cast<double>(non_transition_total);
    }

    return result;
}

} // namespace hdc_systemc
