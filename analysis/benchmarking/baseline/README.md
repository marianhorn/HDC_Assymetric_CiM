# Baseline Benchmark Plots

# Baseline Benchmark Plots

Scripts live in `scripts/`. Generated PNG/HTML plots go to `plots/`.
Averaged CSV outputs go to `results/`.

Generate mean test accuracy over all available seeds versus vector dimension
for 40 quantization levels:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_dimension.py
```

Use a different quantization level count with:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_dimension.py --num-levels 30
```

Limit the shown vector dimensions or vertical accuracy range with:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_dimension.py --min-dimension 1000 --max-dimension 10000 --y-min 0.5 --y-max 1.0
```

Generate mean test accuracy versus quantization levels for one vector
dimension:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_num_levels.py
```

Use a different vector dimension with:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_num_levels.py --vector-dimension 20000
```

Limit the shown quantization level range, for example levels 20 to 50:

```sh
python3 analysis/benchmarking/baseline/scripts/plot_accuracy_by_num_levels.py --min-num-levels 20 --max-num-levels 50
```

Each script prints every selected configuration and its seed coverage. It opens
the figure interactively and saves a high-resolution PNG plot, an HTML plot,
and the averaged CSV. Use `--no-show` for headless runs.
