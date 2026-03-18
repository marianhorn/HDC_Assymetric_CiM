import argparse
import os
from glob import glob


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_ROOT = os.path.join(REPO_ROOT, "foot", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "plots")


def load_csv_matrix(path):
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    return data


def load_csv_labels(path):
    labels = np.genfromtxt(path, delimiter=",", skip_header=1)
    if labels.ndim == 0:
        labels = np.array([labels], dtype=int)
    return labels.astype(int)


def load_dataset(dataset_dir, split):
    if split == "training":
        emg_path = os.path.join(dataset_dir, "training_emg.csv")
        label_path = os.path.join(dataset_dir, "training_labels.csv")
        emg = load_csv_matrix(emg_path)
        labels = load_csv_labels(label_path)
    elif split == "testing":
        emg_path = os.path.join(dataset_dir, "testing_emg.csv")
        label_path = os.path.join(dataset_dir, "testing_labels.csv")
        emg = load_csv_matrix(emg_path)
        labels = load_csv_labels(label_path)
    else:
        emg_train = load_csv_matrix(os.path.join(dataset_dir, "training_emg.csv"))
        labels_train = load_csv_labels(os.path.join(dataset_dir, "training_labels.csv"))
        emg_test = load_csv_matrix(os.path.join(dataset_dir, "testing_emg.csv"))
        labels_test = load_csv_labels(os.path.join(dataset_dir, "testing_labels.csv"))
        emg = np.vstack([emg_train, emg_test])
        labels = np.concatenate([labels_train, labels_test])

    if emg.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"Sample/label mismatch in {dataset_dir}: emg={emg.shape[0]}, labels={labels.shape[0]}"
        )
    return emg, labels


def empirical_cdf(values):
    x = np.sort(values)
    n = x.size
    if n == 0:
        return np.array([]), np.array([])
    y = np.arange(1, n + 1, dtype=float) / float(n)
    return x, y


def plot_dataset_feature_grid(dataset_name, emg, labels, bins, show, out_path):
    num_samples, num_features = emg.shape
    classes = sorted(np.unique(labels))

    ncols = 4
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    cmap = plt.get_cmap("tab10")
    class_colors = {cls: cmap(i % 10) for i, cls in enumerate(classes)}

    legend_handles = []
    legend_labels = []
    legend_seeded = False

    for feature_idx in range(num_features):
        row = feature_idx // ncols
        col = feature_idx % ncols
        ax = axes[row, col]
        ax2 = ax.twinx()

        values_all = emg[:, feature_idx]
        hist_all = ax.hist(
            values_all,
            bins=bins,
            density=True,
            color="lightgray",
            edgecolor="none",
            alpha=0.55,
            label="all-hist",
        )

        cdf_x_all, cdf_y_all = empirical_cdf(values_all)
        cdf_all, = ax2.plot(
            cdf_x_all,
            cdf_y_all,
            color="black",
            linewidth=1.4,
            label="all-cdf",
        )

        class_hist_handles = []
        class_cdf_handles = []
        for cls in classes:
            class_values = values_all[labels == cls]
            if class_values.size == 0:
                continue

            hist_vals, edges = np.histogram(class_values, bins=bins, density=True)
            centers = 0.5 * (edges[:-1] + edges[1:])
            hist_line, = ax.plot(
                centers,
                hist_vals,
                color=class_colors[cls],
                linewidth=1.0,
                alpha=0.95,
                label=f"class {cls} hist",
            )
            class_hist_handles.append((cls, hist_line))

            cdf_x, cdf_y = empirical_cdf(class_values)
            cdf_line, = ax2.plot(
                cdf_x,
                cdf_y,
                color=class_colors[cls],
                linewidth=0.9,
                linestyle="--",
                alpha=0.95,
                label=f"class {cls} cdf",
            )
            class_cdf_handles.append((cls, cdf_line))

        ax.set_title(f"Feature {feature_idx}", fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.set_xlabel("Value")
        ax.set_ylabel("Hist density")
        ax2.set_ylim(0.0, 1.0)
        ax2.set_ylabel("CDF")

        if not legend_seeded:
            legend_handles.append(hist_all[2][0])
            legend_labels.append("All histogram")
            legend_handles.append(cdf_all)
            legend_labels.append("All CDF")

            for cls, h in class_hist_handles:
                legend_handles.append(h)
                legend_labels.append(f"Class {cls} histogram")
            for cls, h in class_cdf_handles:
                legend_handles.append(h)
                legend_labels.append(f"Class {cls} CDF")
            legend_seeded = True

    total_axes = nrows * ncols
    for idx in range(num_features, total_axes):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    fig.suptitle(
        f"{dataset_name}: Feature Value Distributions and CDFs "
        f"(samples={num_samples}, features={num_features})",
        fontsize=14,
    )
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.005),
            ncol=min(4, len(legend_handles)),
            fontsize=9,
            frameon=False,
        )

    fig.tight_layout(rect=[0, 0.01, 1, 0.975])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze foot EMG datasets and create one figure per dataset with "
            "per-feature histograms (overall and per-class) and CDF overlays."
        )
    )
    parser.add_argument(
        "--split",
        default="combined",
        choices=["combined", "training", "testing"],
        help="Which split to analyze (default: combined = training+testing).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Histogram bin count for all features (default: 40).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively in addition to saving.",
    )
    args = parser.parse_args()

    global np, plt
    try:
        import numpy as np_mod
        import matplotlib.pyplot as plt_mod
    except Exception as exc:
        raise RuntimeError(
            "Missing Python dependencies. Install with: pip install numpy matplotlib"
        ) from exc
    np = np_mod
    plt = plt_mod

    dataset_dirs = sorted(glob(os.path.join(DATA_ROOT, "dataset[0-9][0-9]")))
    if not dataset_dirs:
        raise RuntimeError(f"No datasetXX folders found in {DATA_ROOT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Data root: {DATA_ROOT}")
    print(f"Split: {args.split}")
    print(f"Bins: {args.bins}")
    print("")

    for ds_dir in dataset_dirs:
        dataset_name = os.path.basename(ds_dir)
        emg, labels = load_dataset(ds_dir, args.split)
        out_name = f"{dataset_name}_feature_hist_cdf_{args.split}.png"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        print(
            f"Plotting {dataset_name}: samples={emg.shape[0]}, features={emg.shape[1]}, classes={sorted(np.unique(labels))}"
        )
        plot_dataset_feature_grid(
            dataset_name=dataset_name,
            emg=emg,
            labels=labels,
            bins=args.bins,
            show=args.show,
            out_path=out_path,
        )
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
