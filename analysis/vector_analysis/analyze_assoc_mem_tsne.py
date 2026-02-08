import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required for t-SNE. Install with: pip install scikit-learn"
    ) from exc

PRE_FILE = "assocMemPreopt.csv"
POST_FILE = "assocMemPostopt.csv"
RANDOM_STATE = 0


def load_assoc_vectors(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    X = np.loadtxt(path, delimiter=",", comments="#")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pre_path = os.path.join(base_dir, PRE_FILE)
    post_path = os.path.join(base_dir, POST_FILE)

    pre = load_assoc_vectors(pre_path)
    post = load_assoc_vectors(post_path)

    if pre.shape[0] != post.shape[0]:
        print("Warning: pre/post have different number of classes.")

    num_classes = min(pre.shape[0], post.shape[0])
    pre = pre[:num_classes]
    post = post[:num_classes]

    combined = np.vstack([pre, post])
    n_samples = combined.shape[0]
    perplexity = min(5, n_samples - 1)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_STATE)
    emb = tsne.fit_transform(combined)

    pre_emb = emb[:num_classes]
    post_emb = emb[num_classes:]

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    axes[0].set_title("Associative Memory (Pre-opt)")
    axes[1].set_title("Associative Memory (Post-opt)")

    for cls in range(num_classes):
        axes[0].scatter(pre_emb[cls, 0], pre_emb[cls, 1], color=colors[cls], label=f"Class {cls}")
        axes[1].scatter(post_emb[cls, 0], post_emb[cls, 1], color=colors[cls], label=f"Class {cls}")

    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", title="Class")
    fig.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


if __name__ == "__main__":
    main()
