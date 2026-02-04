import os
import numpy as np
import matplotlib.pyplot as plt

# =============================
# USER-CONFIGURABLE CONSTANTS
# =============================
DIM = 10000
NUM_LEVELS = 101

CIM_NAIVE_FILE = "./ds3/cim_naive.csv"
CIM_GA_FILE    = "./ds3/cim_ga.csv"

# =============================
# HELPERS
# =============================
def load_cim_csv(path, expected_rows=NUM_LEVELS, expected_cols=DIM):
    """
    Loads CiM vectors from a comma-separated CSV file.
    Expected shape: [expected_rows, expected_cols].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    X = np.loadtxt(path, delimiter=",")

    # If a single row is loaded as 1D (not expected here), reshape.
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape != (expected_rows, expected_cols):
        raise ValueError(
            f"Unexpected shape in {path}: got {X.shape}, "
            f"expected {(expected_rows, expected_cols)}"
        )

    return X


def cosine_similarity_matrix(V):
    """
    V: [L, D]
    Returns cosine similarity matrix [L, L].
    """
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    Vn = V / norms
    return Vn @ Vn.T


def consecutive_cosine_distances(V):
    """
    Returns array of length L-1:
    dist[l] = 1 - cos(V[l], V[l+1])
    """
    norms = np.linalg.norm(V, axis=1) + 1e-12
    dots = np.sum(V[:-1] * V[1:], axis=1)
    cos = dots / (norms[:-1] * norms[1:])
    return 1.0 - cos


def classical_mds_from_distance(D, out_dim=2):
    """
    Classical (metric) MDS.
    Input: distance matrix D [n,n]
    Output: coordinates Y [n, out_dim]
    """
    n = D.shape[0]
    D2 = D ** 2

    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J

    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Use only the top out_dim eigenpairs; clip negatives for stability
    eigvals = np.clip(eigvals[:out_dim], a_min=0.0, a_max=None)
    Y = eigvecs[:, :out_dim] * np.sqrt(eigvals + 1e-12)
    return Y
def cosine_distance(u, v):
    """
    Returns cosine distance: 1 - cos(u, v)
    """
    num = np.dot(u, v)
    den = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return 1.0 - (num / den)

# =============================
# ANALYSIS
# =============================
def analyze_cim(name, V):
    """
    Produces:
      - Adjacent cosine distance plot
      - Similarity heatmap
      - MDS embedding of levels
    """
    d_adj = consecutive_cosine_distances(V)

    S = cosine_similarity_matrix(V)
    D = 1.0 - S  # cosine distance

    Y = classical_mds_from_distance(D, out_dim=2)

    # Plot 1: adjacent distances
    plt.figure()
    plt.plot(np.arange(NUM_LEVELS - 1), d_adj)
    plt.title(f"{name}: Adjacent level cosine distance")
    plt.xlabel("Level l (distance between l and l+1)")
    plt.ylabel("1 - cosine similarity")
    plt.grid(True)

    # Plot 2: similarity heatmap
    plt.figure()
    plt.imshow(S, aspect="auto")
    plt.title(f"{name}: Level cosine similarity matrix")
    plt.xlabel("Level")
    plt.ylabel("Level")
    plt.colorbar(label="cosine similarity")

    # Plot 3: MDS embedding path
    plt.figure()
    plt.plot(Y[:, 0], Y[:, 1], marker="o")
    for l in range(NUM_LEVELS):
        if l % 10 == 0 or l == NUM_LEVELS - 1:
            plt.text(Y[l, 0], Y[l, 1], str(l), fontsize=9)
    plt.title(f"{name}: Classical MDS of levels (cosine distance)")
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.grid(True)

    return d_adj


def main():
    V_naive = load_cim_csv(CIM_NAIVE_FILE)
    V_ga    = load_cim_csv(CIM_GA_FILE)

    # Optional sanity check (uncomment if desired)
    # print("Naive sample unique:", np.unique(V_naive[:2, :50]))
    # print("GA sample unique:", np.unique(V_ga[:2, :50]))

    d_naive = analyze_cim("Naive CiM", V_naive)
    d_ga    = analyze_cim("GA CiM", V_ga)
    # Distance between min and max level
    dist_naive_min_max = cosine_distance(V_naive[0], V_naive[NUM_LEVELS - 1])
    dist_ga_min_max    = cosine_distance(V_ga[0],    V_ga[NUM_LEVELS - 1])

    print("Cosine distance between min and max level:")
    print(f"  Naive CiM: {dist_naive_min_max:.6f}")
    print(f"  GA CiM:    {dist_ga_min_max:.6f}")
    # Comparison plot
    plt.figure()
    plt.plot(np.arange(NUM_LEVELS - 1), d_naive, label="Naive")
    plt.plot(np.arange(NUM_LEVELS - 1), d_ga, label="GA")
    plt.title("Adjacent level cosine distance: Naive vs GA")
    plt.xlabel("Level l (distance between l and l+1)")
    plt.ylabel("1 - cosine similarity")
    plt.legend()
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()
