import os
import numpy as np
import matplotlib.pyplot as plt

# =============================
# USER-CONFIGURABLE CONSTANTS
# =============================
DEFAULT_DIM = 1000
DEFAULT_NUM_LEVELS = 31
DEFAULT_NUM_FEATURES = 32

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CIM_NAIVE_FILE = os.path.join(BASE_DIR, "./item_mem_naive.csv")
CIM_OPT_FILE   = os.path.join(BASE_DIR, "./item_mem_optimized.csv")

# Set to True/False to override auto-detection. Leave as None for auto.
FORCE_BINARY_MODE = None

# Comparison plot mode:
#   "single" -> plot one feature only (recommended)
#   "all"    -> plot all features (thin lines) + bold mean
COMPARE_MODE = "all"
COMPARE_FEATURE_INDEX = 5          # used when COMPARE_MODE="single"
ALL_FEATURE_ALPHA = 0.15           # used when COMPARE_MODE="all"
ALL_FEATURE_LINEWIDTH = 0.8        # used when COMPARE_MODE="all"

# =============================
# HELPERS
# =============================
def parse_csv_header(path):
    header = {}
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    if not first.startswith("#"):
        return header
    parts = first[1:].strip().split(",")
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            header[key] = int(value)
    return header


def load_precomp_item_mem(path,
                          default_levels=DEFAULT_NUM_LEVELS,
                          default_features=DEFAULT_NUM_FEATURES,
                          default_dim=DEFAULT_DIM):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    header = parse_csv_header(path)
    num_levels = header.get("num_levels", default_levels)
    num_features = header.get("num_features", default_features)
    num_vectors = header.get("num_vectors", None)
    dim = header.get("dimension", default_dim)

    X = np.loadtxt(path, delimiter=",", comments="#")
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if dim is None or dim <= 0:
        dim = X.shape[1]

    if num_levels is None or num_features is None:
        if num_levels is None and num_features is not None:
            num_levels = X.shape[0] // num_features
        elif num_features is None and num_levels is not None:
            num_features = X.shape[0] // num_levels
        else:
            raise ValueError("Cannot infer num_levels/num_features; set defaults or include header.")

    expected = num_levels * num_features
    if X.shape[0] != expected:
        if num_vectors is not None and X.shape[0] == num_vectors:
            expected = num_vectors
        else:
            raise ValueError(
                f"Unexpected rows in {path}: got {X.shape[0]}, expected {expected} "
                f"(levels={num_levels}, features={num_features})"
            )

    if X.shape[1] != dim:
        print(f"Warning: dimension mismatch in {path}: header {dim}, file {X.shape[1]}")
        dim = X.shape[1]

    V = X.reshape((num_levels, num_features, dim))
    return V


def cosine_similarity_matrix(V):
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    Vn = V / norms
    return Vn @ Vn.T

def hamming_similarity_matrix(V):
    diff = V[:, None, :] != V[None, :, :]
    dist = diff.mean(axis=2)
    return 1.0 - dist


def consecutive_cosine_distances(V):
    norms = np.linalg.norm(V, axis=1) + 1e-12
    dots = np.sum(V[:-1] * V[1:], axis=1)
    cos = dots / (norms[:-1] * norms[1:])
    return 1.0 - cos

def consecutive_hamming_distances(V):
    return (V[:-1] != V[1:]).mean(axis=1)


def classical_mds_from_distance(D, out_dim=2):
    n = D.shape[0]
    D2 = D ** 2

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    eigvals = np.clip(eigvals[:out_dim], a_min=0.0, a_max=None)
    Y = eigvecs[:, :out_dim] * np.sqrt(eigvals + 1e-12)
    return Y


def cosine_distance(u, v):
    num = np.dot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return 1.0 - (num / den)

def hamming_distance(u, v):
    return np.mean(u != v)

def is_binary_vectors(V):
    return np.all((V == 0) | (V == 1))

def similarity_matrix(V, binary_mode):
    if binary_mode:
        return hamming_similarity_matrix(V)
    return cosine_similarity_matrix(V)

def consecutive_distances(V, binary_mode):
    if binary_mode:
        return consecutive_hamming_distances(V)
    return consecutive_cosine_distances(V)

def pair_distance(u, v, binary_mode):
    if binary_mode:
        return hamming_distance(u, v)
    return cosine_distance(u, v)

# =============================
# ANALYSIS
# =============================

def analyze_precomp_cim(name, V, binary_mode):
    num_levels, num_features, _ = V.shape

    d_adj_all = np.zeros((num_features, num_levels - 1))
    sim_sum = np.zeros((num_levels, num_levels))

    for f in range(num_features):
        Vf = V[:, f, :]
        d_adj_all[f] = consecutive_distances(Vf, binary_mode)
        sim_sum += similarity_matrix(Vf, binary_mode)

    d_adj_mean = d_adj_all.mean(axis=0)
    d_adj_std = d_adj_all.std(axis=0)

    S = sim_sum / num_features
    D = 1.0 - S
    Y = classical_mds_from_distance(D, out_dim=2)

    distance_label = "Hamming distance" if binary_mode else "1 - cosine similarity"
    sim_label = "Hamming similarity" if binary_mode else "cosine similarity"
    title_metric = "Hamming" if binary_mode else "cosine"

    # Plot 1: adjacent distances (mean+/-std)
    plt.figure()
    x = np.arange(num_levels - 1)
    plt.plot(x, d_adj_mean, label=f"{name} mean")
    plt.fill_between(x, d_adj_mean - d_adj_std, d_adj_mean + d_adj_std, alpha=0.2, label=f"{name} +/-1sigma")
    plt.title(f"{name}: Adjacent level {title_metric} distance (mean across features)")
    plt.xlabel("Level l (distance between l and l+1)")
    plt.ylabel(distance_label)
    plt.legend()
    plt.grid(True)

    # Plot 2: similarity heatmap (mean)
    plt.figure()
    plt.imshow(S, aspect="auto")
    plt.title(f"{name}: Level {sim_label} matrix (mean across features)")
    plt.xlabel("Level")
    plt.ylabel("Level")
    plt.colorbar(label=sim_label)

    # Plot 3: MDS embedding path
    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], marker="o")
    for l in range(num_levels):
        if l % 10 == 0 or l == num_levels - 1:
            plt.text(Y[l, 0], Y[l, 1], str(l), fontsize=9)
    plt.title(f"{name}: Classical MDS of levels (mean {title_metric} distance)")
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.grid(True)

    return d_adj_all, d_adj_mean, d_adj_std




def plot_adjacent_comparison_per_feature(V_naive, V_opt, binary_mode, mode="single", feature_index=0):
    """
    Adds a comparison plot for adjacent distances without aggregating away features.

    mode:
      - "single": plot one feature (most readable)
      - "all": plot all features as faint lines + bold mean
    """
    L, F, _ = V_naive.shape
    x = np.arange(L - 1)

    # Compute adjacent distance curves for each feature
    d_naive_all = np.zeros((F, L - 1))
    d_opt_all   = np.zeros((F, L - 1))
    for f in range(F):
        d_naive_all[f] = consecutive_distances(V_naive[:, f, :], binary_mode)
        d_opt_all[f]   = consecutive_distances(V_opt[:, f, :], binary_mode)

    plt.figure()

    if mode == "single":
        f = int(feature_index)
        if f < 0 or f >= F:
            raise ValueError(f"COMPARE_FEATURE_INDEX out of range: {f} (num_features={F})")

        plt.plot(x, d_naive_all[f], label=f"Naive (feature {f})")
        plt.plot(x, d_opt_all[f],   label=f"Optimized (feature {f})")

        metric = "Hamming" if binary_mode else "cosine"
        plt.title(f"Adjacent {metric} distance (feature {f}): Naive vs Optimized")

    elif mode == "all":
        # plot all features lightly
        for f in range(F):
            plt.plot(x, d_naive_all[f], alpha=ALL_FEATURE_ALPHA, linewidth=ALL_FEATURE_LINEWIDTH)
        for f in range(F):
            plt.plot(x, d_opt_all[f], alpha=ALL_FEATURE_ALPHA, linewidth=ALL_FEATURE_LINEWIDTH)

        # overlay means boldly for readability
        plt.plot(x, d_naive_all.mean(axis=0), linewidth=2.5, label="Naive mean")
        plt.plot(x, d_opt_all.mean(axis=0),   linewidth=2.5, label="Optimized mean")

        metric = "Hamming" if binary_mode else "cosine"
        plt.title(f"Adjacent {metric} distance (all features): Naive vs Optimized")

    else:
        raise ValueError("mode must be 'single' or 'all'")

    plt.xlabel("Level l (distance between l and l+1)")
    plt.ylabel("Hamming distance" if binary_mode else "1 - cosine similarity")
    plt.legend()
    plt.grid(True)



def main():
    V_naive = load_precomp_item_mem(CIM_NAIVE_FILE)
    if FORCE_BINARY_MODE is None:
        binary_mode = is_binary_vectors(V_naive)
    else:
        binary_mode = bool(FORCE_BINARY_MODE)
    V_opt = None
    if os.path.exists(CIM_OPT_FILE):
        V_opt = load_precomp_item_mem(CIM_OPT_FILE)
    else:
        print(f"Warning: optimized file not found at {CIM_OPT_FILE}.")

    d_naive_all, d_naive_mean, d_naive_std = analyze_precomp_cim("Naive CiM", V_naive, binary_mode)

    # Distance between min and max level (mean across features)
    num_levels, num_features, _ = V_naive.shape
    dist_naive = [pair_distance(V_naive[0, f], V_naive[num_levels - 1, f], binary_mode) for f in range(num_features)]
    metric_label = "Hamming" if binary_mode else "Cosine"
    print(f"{metric_label} distance between min and max level (mean across features):")
    print(f"  Naive CiM: {float(np.mean(dist_naive)):.6f}")

    if V_opt is not None:
        d_opt_all, d_opt_mean, d_opt_std = analyze_precomp_cim("Optimized CiM", V_opt, binary_mode)

        num_levels_opt, num_features_opt, _ = V_opt.shape
        dist_opt = [pair_distance(V_opt[0, f], V_opt[num_levels_opt - 1, f], binary_mode) for f in range(num_features_opt)]
        print(f"  Optimized CiM: {float(np.mean(dist_opt)):.6f}")
        # Existing comparison plot (mean+/-std)
        plt.figure()
        x = np.arange(len(d_naive_mean))
        plt.plot(x, d_naive_mean, label="Naive mean")
        plt.fill_between(x, d_naive_mean - d_naive_std, d_naive_mean + d_naive_std, alpha=0.2)
        plt.plot(x, d_opt_mean, label="Optimized mean")
        plt.fill_between(x, d_opt_mean - d_opt_std, d_opt_mean + d_opt_std, alpha=0.2)
        metric = "Hamming" if binary_mode else "cosine"
        plt.title(f"Adjacent level {metric} distance: Naive vs Optimized (mean+/-std)")
        plt.xlabel("Level l (distance between l and l+1)")
        plt.ylabel("Hamming distance" if binary_mode else "1 - cosine similarity")
        plt.legend()
        plt.grid(True)

        # NEW: comparison without averaging away features
        plot_adjacent_comparison_per_feature(
            V_naive, V_opt,
            binary_mode,
            mode=COMPARE_MODE,
            feature_index=COMPARE_FEATURE_INDEX
        )

    plt.show()


if __name__ == "__main__":
    main()
