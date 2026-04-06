import argparse
import csv
import os
from collections import OrderedDict

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, 'results.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'plots')
SUMMARY_PATH = os.path.join(BASE_DIR, 'ga_quantizer_plot_summary.csv')

PHASES = OrderedDict([
    ('uniform-test', 'Uniform quantizer'),
    ('uniform-ga-test', 'Uniform quantizer + CiM GA'),
    ('preopt-test', 'GA quantizer, no CiM GA'),
    ('postopt-test', 'GA quantizer + CiM GA'),
])

PHASE_STYLES = {
    'uniform-test': {'color': '#666666', 'linestyle': '--', 'marker': 'o'},
    'uniform-ga-test': {'color': '#d62728', 'linestyle': '-', 'marker': 's'},
    'preopt-test': {'color': '#1f77b4', 'linestyle': '-', 'marker': '^'},
    'postopt-test': {'color': '#2ca02c', 'linestyle': '-', 'marker': 'D'},
}


def parse_info(info_text):
    result = {}
    for chunk in info_text.split(','):
        if '=' not in chunk:
            continue
        key, value = chunk.split('=', 1)
        result[key] = value
    return result


def load_overall_rows(results_path):
    if not os.path.exists(results_path):
        raise FileNotFoundError(f'Missing results file: {results_path}')

    grouped = {}
    with open(results_path, 'r', encoding='utf-8', newline='') as results_file:
        reader = csv.DictReader(results_file)
        for row in reader:
            info = parse_info(row['info'])
            if info.get('scope') != 'overall':
                continue
            phase = info.get('phase')
            if phase not in PHASES:
                continue

            key = (int(row['num_levels']), int(row['vector_dimension']))
            if key not in grouped:
                grouped[key] = {
                    'num_levels': key[0],
                    'vector_dimension': key[1],
                    **{phase_name: None for phase_name in PHASES},
                }
            grouped[key][phase] = float(row['overall_accuracy']) * 100.0

    rows = [grouped[key] for key in sorted(grouped)]
    if not rows:
        raise RuntimeError('No overall GA-quantizer rows found in results.csv.')
    return rows


def write_summary_csv(rows):
    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, 'w', encoding='utf-8', newline='') as summary_file:
        writer = csv.DictWriter(
            summary_file,
            fieldnames=['num_levels', 'vector_dimension'] + list(PHASES.keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_sweep(rows, *, x_key, x_label, fixed_key, fixed_value, title_suffix, filename):
    if plt is None:
        return None

    subset = [row for row in rows if row[fixed_key] == fixed_value]
    subset.sort(key=lambda item: item[x_key])
    if not subset:
        raise RuntimeError(f'No rows found for {fixed_key}={fixed_value}.')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    xs = [row[x_key] for row in subset]
    for phase, label in PHASES.items():
        ys = [row[phase] for row in subset]
        style = PHASE_STYLES[phase]
        ax.plot(xs, ys, label=label, linewidth=2.0, markersize=6, **style)

    ax.set_xlabel(x_label)
    ax.set_ylabel('Overall test accuracy (%)')
    ax.set_title(title_suffix)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description='Plot GA-quantizer experiment results for the two configured sweeps.'
    )
    parser.add_argument(
        '--fixed-vector-dimension',
        type=int,
        default=512,
        help='Vector dimension for sweep 1 over NUM_LEVELS (default: 512).',
    )
    parser.add_argument(
        '--fixed-num-levels',
        type=int,
        default=20,
        help='NUM_LEVELS for sweep 2 over VECTOR_DIMENSION (default: 20).',
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show plots interactively in addition to saving them.',
    )
    args = parser.parse_args()

    rows = load_overall_rows(RESULTS_PATH)
    write_summary_csv(rows)

    if plt is None:
        print('matplotlib is not installed. Wrote the summary CSV but skipped plots.')
        print(f'Summary CSV: {SUMMARY_PATH}')
        return

    sweep1_path = plot_sweep(
        rows,
        x_key='num_levels',
        x_label='NUM_LEVELS',
        fixed_key='vector_dimension',
        fixed_value=args.fixed_vector_dimension,
        title_suffix=(
            f'GA quantizer experiment: overall test accuracy vs NUM_LEVELS '
            f'(VECTOR_DIMENSION={args.fixed_vector_dimension})'
        ),
        filename=f'ga_quantizer_vs_num_levels_vd{args.fixed_vector_dimension}.png',
    )
    sweep2_path = plot_sweep(
        rows,
        x_key='vector_dimension',
        x_label='VECTOR_DIMENSION',
        fixed_key='num_levels',
        fixed_value=args.fixed_num_levels,
        title_suffix=(
            f'GA quantizer experiment: overall test accuracy vs VECTOR_DIMENSION '
            f'(NUM_LEVELS={args.fixed_num_levels})'
        ),
        filename=f'ga_quantizer_vs_vector_dimension_nl{args.fixed_num_levels}.png',
    )

    if args.show:
        for image_path in [sweep1_path, sweep2_path]:
            img = plt.imread(image_path)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.axis('off')
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    print(f'Summary CSV: {SUMMARY_PATH}')
    print(f'Sweep 1 plot: {sweep1_path}')
    print(f'Sweep 2 plot: {sweep2_path}')


if __name__ == '__main__':
    main()
