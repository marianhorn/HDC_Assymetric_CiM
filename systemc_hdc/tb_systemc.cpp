#include <cstdlib>
#include <iostream>
#include "hdc_systemc.h"

using namespace hdc_systemc;

namespace {

static const int TRAINING_SAMPLES = 12;
static const int EVAL_SAMPLES = 12;

void set_bit(hv_t &hv, int index, bool value) {
    hv[index] = value ? sc_dt::SC_LOGIC_1 : sc_dt::SC_LOGIC_0;
}

bool get_bit(const hv_t &hv, int index) {
    return hv[index].to_bool();
}

void clear_hv(hv_t &hv) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        hv[d] = sc_dt::SC_LOGIC_0;
    }
}

bool hv_equal(const hv_t &lhs, const hv_t &rhs) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        if (lhs[d] != rhs[d]) {
            return false;
        }
    }
    return true;
}

void xor_hv(const hv_t &lhs, const hv_t &rhs, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        set_bit(dst, d, get_bit(lhs, d) ^ get_bit(rhs, d));
    }
}

void permute_right(const hv_t &src, hv_t &dst) {
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        const int out_index = (d + 1) % VECTOR_DIMENSION;
        dst[out_index] = src[d];
    }
}

void build_demo_cim(hv_t *flat_cim) {
    for (int level = 0; level < NUM_LEVELS; ++level) {
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            hv_t &entry = flat_cim[(level * NUM_FEATURES) + feature];
            clear_hv(entry);
            for (int d = 0; d < VECTOR_DIMENSION; ++d) {
                const unsigned value =
                    static_cast<unsigned>((level * 29) + (feature * 17) + (d * 7) + (d >> 2));
                set_bit(entry, d, (value & 1u) != 0u);
            }
        }
    }
}

void build_demo_dataset(level_t *training_levels,
                        int *training_labels,
                        level_t *eval_levels,
                        int *eval_labels) {
    const int label_pattern[TRAINING_SAMPLES] = {0, 0, 0, 1, 1, 1, 2, 2, 2, 1, 1, 1};
    const int eval_pattern[EVAL_SAMPLES] = {0, 0, 0, 1, 2, 1, 2, 2, 2, 1, 1, 1};

    for (int sample = 0; sample < TRAINING_SAMPLES; ++sample) {
        training_labels[sample] = label_pattern[sample];
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const unsigned raw_level =
                static_cast<unsigned>((label_pattern[sample] * 9) + sample + (feature * 3));
            training_levels[(sample * NUM_FEATURES) + feature] =
                static_cast<unsigned>(raw_level % NUM_LEVELS);
        }
    }

    for (int sample = 0; sample < EVAL_SAMPLES; ++sample) {
        eval_labels[sample] = eval_pattern[sample];
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const unsigned raw_level =
                static_cast<unsigned>((eval_pattern[sample] * 11) + sample + (feature * 5));
            eval_levels[(sample * NUM_FEATURES) + feature] =
                static_cast<unsigned>(raw_level % NUM_LEVELS);
        }
    }
}

void reference_encode_timestamp(const hv_t *flat_cim, const level_t *frame_levels, hv_t &encoded) {
    const unsigned threshold = NUM_FEATURES / 2;
    for (int d = 0; d < VECTOR_DIMENSION; ++d) {
        unsigned ones = 0;
        for (int feature = 0; feature < NUM_FEATURES; ++feature) {
            const hv_t &entry = flat_cim[(frame_levels[feature].to_uint() * NUM_FEATURES) + feature];
            if (get_bit(entry, d)) {
                ++ones;
            }
        }
        set_bit(encoded, d, ones >= threshold);
    }
}

void reference_encode_ngram(const hv_t *flat_cim, const level_t *level_window, hv_t &encoded) {
    reference_encode_timestamp(flat_cim, level_window, encoded);

    hv_t current;
    hv_t permuted;
    for (int i = 1; i < N_GRAM_SIZE; ++i) {
        reference_encode_timestamp(flat_cim, &level_window[i * NUM_FEATURES], current);
        permute_right(encoded, permuted);
        xor_hv(permuted, current, encoded);
    }
}

void reference_train(const hv_t *flat_cim,
                     const level_t *training_levels,
                     const int *training_labels,
                     train_counter_t class_bit_counts[NUM_CLASSES][VECTOR_DIMENSION],
                     train_counter_t class_counts[NUM_CLASSES],
                     hv_t class_vectors[NUM_CLASSES]) {
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        class_counts[class_id] = 0;
        clear_hv(class_vectors[class_id]);
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            class_bit_counts[class_id][d] = 0;
        }
    }

    hv_t encoded;
    for (int j = 0; j < TRAINING_SAMPLES - N_GRAM_SIZE; ++j) {
        if (training_labels[j] == training_labels[j + N_GRAM_SIZE - 1]) {
            const int class_id = training_labels[j];
            reference_encode_ngram(flat_cim, &training_levels[j * NUM_FEATURES], encoded);
            if (class_id >= 0 && class_id < NUM_CLASSES) {
                for (int d = 0; d < VECTOR_DIMENSION; ++d) {
                    if (get_bit(encoded, d)) {
                        class_bit_counts[class_id][d] = class_bit_counts[class_id][d] + 1;
                    }
                }
                class_counts[class_id] = class_counts[class_id] + 1;
            }
        } else {
            j += (N_GRAM_SIZE - 1);
        }
    }

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        const train_counter_t threshold = class_counts[class_id] / 2;
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            set_bit(class_vectors[class_id], d, class_bit_counts[class_id][d] >= threshold);
        }
    }
}

int reference_predict(const hv_t class_vectors[NUM_CLASSES], const hv_t &query) {
    unsigned best_distance = VECTOR_DIMENSION + 1;
    int best_class = 0;
    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        unsigned distance = 0;
        for (int d = 0; d < VECTOR_DIMENSION; ++d) {
            if (get_bit(query, d) != get_bit(class_vectors[class_id], d)) {
                ++distance;
            }
        }
        if (distance < best_distance) {
            best_distance = distance;
            best_class = class_id;
        }
    }
    return best_class;
}

bool run_unit_checks(HdcTop &top,
                     const hv_t *flat_cim,
                     const level_t *training_levels,
                     const int *training_labels,
                     const level_t *eval_levels) {
    hv_t reference_timestamp;
    hv_t systemc_timestamp;
    reference_encode_timestamp(flat_cim, &training_levels[0], reference_timestamp);
    top.encode_timestamp(&training_levels[0], systemc_timestamp);
    if (!hv_equal(reference_timestamp, systemc_timestamp)) {
        std::cerr << "Timestamp encoder mismatch." << std::endl;
        return false;
    }

    hv_t reference_ngram;
    hv_t systemc_ngram;
    reference_encode_ngram(flat_cim, &training_levels[0], reference_ngram);
    top.encode_ngram(&training_levels[0], systemc_ngram);
    if (!hv_equal(reference_ngram, systemc_ngram)) {
        std::cerr << "Fusion n-gram encoder mismatch." << std::endl;
        return false;
    }

    train_counter_t ref_bit_counts[NUM_CLASSES][VECTOR_DIMENSION];
    train_counter_t ref_class_counts[NUM_CLASSES];
    hv_t ref_class_vectors[NUM_CLASSES];
    reference_train(flat_cim, training_levels, training_labels, ref_bit_counts, ref_class_counts, ref_class_vectors);

    top.reset_assoc_mem();
    top.train_dataset(training_levels, training_labels, TRAINING_SAMPLES);
    top.finalize_assoc_mem();

    for (int class_id = 0; class_id < NUM_CLASSES; ++class_id) {
        if (!hv_equal(ref_class_vectors[class_id], top.get_class_vector(static_cast<unsigned>(class_id)))) {
            std::cerr << "Associative-memory finalize mismatch for class " << class_id << "." << std::endl;
            return false;
        }
        if (ref_class_counts[class_id] != top.get_class_vector_count(static_cast<unsigned>(class_id))) {
            std::cerr << "Associative-memory count mismatch for class " << class_id << "." << std::endl;
            return false;
        }
    }

    hv_t eval_query;
    reference_encode_ngram(flat_cim, &eval_levels[0], eval_query);
    const int reference_prediction = reference_predict(ref_class_vectors, eval_query);
    const int systemc_prediction = top.predict_ngram(&eval_levels[0]);
    if (reference_prediction != systemc_prediction) {
        std::cerr << "Classifier mismatch." << std::endl;
        return false;
    }

    return true;
}

void print_eval_result(const char *name, const EvaluationResult &result) {
    std::cout << name << " accuracy: " << (result.overall_accuracy * 100.0) << "%" << std::endl;
    std::cout << name << " accuracy excl. transitions: "
              << (result.non_transition_accuracy * 100.0) << "%" << std::endl;
    std::cout << name << " counts: correct=" << result.correct
              << ", wrong=" << result.not_correct
              << ", transitions=" << result.transition_error
              << ", total=" << result.total << std::endl;
}

} // namespace

int sc_main(int, char *[]) {
    hv_t flat_cim[NUM_LEVELS * NUM_FEATURES];
    level_t training_levels[TRAINING_SAMPLES * NUM_FEATURES];
    int training_labels[TRAINING_SAMPLES];
    level_t eval_levels[EVAL_SAMPLES * NUM_FEATURES];
    int eval_labels[EVAL_SAMPLES];

    build_demo_cim(flat_cim);
    build_demo_dataset(training_levels, training_labels, eval_levels, eval_labels);

    HdcTop top("hdc_top");
    top.load_cim(flat_cim);

    if (!run_unit_checks(top, flat_cim, training_levels, training_labels, eval_levels)) {
        return EXIT_FAILURE;
    }

    const EvaluationResult training_result =
        evaluate_dataset(top, training_levels, training_labels, TRAINING_SAMPLES);
    const EvaluationResult eval_result =
        evaluate_dataset(top, eval_levels, eval_labels, EVAL_SAMPLES);

    std::cout << "SystemC HDC pipeline reference checks passed." << std::endl;
    print_eval_result("Training", training_result);
    print_eval_result("Eval", eval_result);

    return EXIT_SUCCESS;
}
