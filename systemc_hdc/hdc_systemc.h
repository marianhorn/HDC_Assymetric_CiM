#ifndef SYSTEMC_HDC_HDC_SYSTEMC_H
#define SYSTEMC_HDC_HDC_SYSTEMC_H

#include <systemc>
#include "config_systemc.h"

namespace hdc_systemc {

constexpr unsigned required_bits(unsigned max_value_exclusive) {
    unsigned bits = 0;
    unsigned limit = 1;
    while (limit < max_value_exclusive) {
        limit <<= 1;
        ++bits;
    }
    return bits == 0 ? 1u : bits;
}

static constexpr unsigned LEVEL_BITS = required_bits(NUM_LEVELS);
static constexpr unsigned CLASS_BITS = required_bits(NUM_CLASSES);
static constexpr unsigned FEATURE_COUNT_BITS = required_bits(NUM_FEATURES + 1);
static constexpr unsigned DISTANCE_BITS = required_bits(VECTOR_DIMENSION + 1);
static constexpr unsigned TRAIN_COUNT_BITS = 32;

typedef sc_dt::sc_uint<LEVEL_BITS> level_t;
typedef sc_dt::sc_uint<CLASS_BITS> class_t;
typedef sc_dt::sc_uint<FEATURE_COUNT_BITS> feature_counter_t;
typedef sc_dt::sc_uint<DISTANCE_BITS> distance_counter_t;
typedef sc_dt::sc_uint<TRAIN_COUNT_BITS> train_counter_t;
typedef sc_dt::sc_bv<VECTOR_DIMENSION> hv_t;

struct EvaluationResult {
    unsigned correct;
    unsigned not_correct;
    unsigned transition_error;
    unsigned total;
    double overall_accuracy;
    double non_transition_accuracy;
    unsigned confusion_matrix[NUM_CLASSES][NUM_CLASSES];
};

SC_MODULE(CiMMemory) {
public:
    SC_CTOR(CiMMemory);

    void clear();
    void load(const hv_t *flat_cim);
    const hv_t &read(level_t level, unsigned feature) const;

private:
    hv_t m_storage[NUM_LEVELS * NUM_FEATURES];
};

SC_MODULE(TimestampEncoder) {
public:
    SC_CTOR(TimestampEncoder);

    void bind_cim(const CiMMemory *cim);
    void encode(const level_t *frame_levels, hv_t &encoded) const;

private:
    const CiMMemory *m_cim;
};

SC_MODULE(FusionNGramEncoder) {
public:
    SC_CTOR(FusionNGramEncoder);

    void bind_timestamp_encoder(const TimestampEncoder *timestamp_encoder);
    void encode_window(const level_t *level_window, hv_t &encoded) const;

private:
    const TimestampEncoder *m_timestamp_encoder;
};

SC_MODULE(AssocMemTrainer) {
public:
    SC_CTOR(AssocMemTrainer);

    void reset();
    void accumulate(int class_id, const hv_t &encoded_ngram);
    void finalize();
    const hv_t &get_class_vector(unsigned class_id) const;
    train_counter_t get_vector_count(unsigned class_id) const;

private:
    train_counter_t m_class_bit_counts[NUM_CLASSES][VECTOR_DIMENSION];
    train_counter_t m_vector_counts[NUM_CLASSES];
    hv_t m_class_vectors[NUM_CLASSES];
};

SC_MODULE(Classifier) {
public:
    SC_CTOR(Classifier);

    void bind_assoc_mem(const AssocMemTrainer *assoc_mem);
    int predict(const hv_t &query) const;
    void compute_distances(const hv_t &query, distance_counter_t *distances) const;

private:
    const AssocMemTrainer *m_assoc_mem;
};

SC_MODULE(HdcTop) {
public:
    SC_CTOR(HdcTop);

    void load_cim(const hv_t *flat_cim);
    void reset_assoc_mem();
    void train_dataset(const level_t *level_data, const int *labels, int num_samples);
    void finalize_assoc_mem();
    void encode_timestamp(const level_t *frame_levels, hv_t &encoded) const;
    void encode_ngram(const level_t *level_window, hv_t &encoded) const;
    int predict_ngram(const level_t *level_window) const;
    void compute_distances_for_ngram(const level_t *level_window, distance_counter_t *distances) const;
    const hv_t &get_class_vector(unsigned class_id) const;
    train_counter_t get_class_vector_count(unsigned class_id) const;

private:
    bool is_window_stable(const int *labels) const;

    CiMMemory m_cim;
    TimestampEncoder m_timestamp_encoder;
    FusionNGramEncoder m_ngram_encoder;
    AssocMemTrainer m_assoc_mem_trainer;
    Classifier m_classifier;
};

int mode_smallest_tie(const int *labels, int size);
EvaluationResult evaluate_dataset(HdcTop &top, const level_t *level_data, const int *labels, int num_samples);

} // namespace hdc_systemc

#endif
