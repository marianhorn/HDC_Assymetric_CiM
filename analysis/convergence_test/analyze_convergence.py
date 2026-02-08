import os
import re
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_NAME = "output.txt"
SHOW_STD = False
SELECTION_LABELS = {
    0: "Pareto",
    1: "Multi (acc-sim)",
    2: "Accuracy",
}

HEADER_RE = re.compile(
    r"NUM_LEVELS=(\d+)\s+VECTOR_DIMENSION=(\d+)\s+GA_SELECTION_MODE=(\d+)\s+GA_INIT_UNIFORM=(\d+)\s+GA_DEFAULT_GENERATIONS=(\d+)"
)
GEN_RE = re.compile(r"GA generation (\d+)/(\d+)")
IND_RE = re.compile(r"individual \d+/\d+ accuracy: ([0-9.]+)%")


def finalize_run(run):
    if not run or not run.get("gen_acc"):
        return None
    max_gen = max(run["gen_acc"].keys())
    means = np.full(max_gen, np.nan)
    for gen, values in run["gen_acc"].items():
        if values:
            means[gen - 1] = float(np.mean(values))
    run["mean_series"] = means
    return run


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, OUTPUT_NAME)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")

    runs = []
    current = None
    current_gen = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            header = HEADER_RE.search(line)
            if header:
                if current:
                    finalized = finalize_run(current)
                    if finalized:
                        runs.append(finalized)
                current = {
                    "num_levels": int(header.group(1)),
                    "vector_dim": int(header.group(2)),
                    "selection_mode": int(header.group(3)),
                    "init_uniform": int(header.group(4)),
                    "generations": int(header.group(5)),
                    "gen_acc": {},
                }
                current_gen = None
                continue

            gen_match = GEN_RE.search(line)
            if gen_match and current:
                current_gen = int(gen_match.group(1))
                current["gen_acc"].setdefault(current_gen, [])
                continue

            ind_match = IND_RE.search(line)
            if ind_match and current and current_gen is not None:
                acc = float(ind_match.group(1))
                current["gen_acc"][current_gen].append(acc)

    if current:
        finalized = finalize_run(current)
        if finalized:
            runs.append(finalized)

    if not runs:
        print("No runs found. Check output.txt format.")
        return

    grouped = {}
    for run in runs:
        key = (run["init_uniform"], run["selection_mode"])
        grouped.setdefault(key, []).append(run["mean_series"])

    for init_uniform in [0, 1]:
        plt.figure()
        has_any = False
        modes_available = sorted({k[1] for k in grouped.keys() if k[0] == init_uniform})
        if not modes_available:
            continue
        print(f"Init_uniform={init_uniform}: modes found {modes_available}")
        for selection_mode in [0, 1, 2]:
            if selection_mode not in modes_available:
                continue
            series_list = grouped[(init_uniform, selection_mode)]
            max_len = max(len(s) for s in series_list)
            stack = np.full((len(series_list), max_len), np.nan)
            for i, s in enumerate(series_list):
                stack[i, :len(s)] = s
            mean = np.nanmean(stack, axis=0)
            std = np.nanstd(stack, axis=0)
            x = np.arange(1, len(mean) + 1)
            label = SELECTION_LABELS.get(selection_mode, f"Mode {selection_mode}")
            plt.plot(x, mean, label=label)
            if SHOW_STD:
                plt.fill_between(x, mean - std, mean + std, alpha=0.2)
            has_any = True
        if has_any:
            title = "Uniform init" if init_uniform == 1 else "Equal init"
            plt.title(f"Mean accuracy per generation ({title})")
            plt.xlabel("Generation")
            plt.ylabel("Mean accuracy (%)")
            plt.grid(True)
            plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
