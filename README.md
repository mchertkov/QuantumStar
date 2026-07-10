# Quantum star: L0/G1 hierarchy code and figures

This repository contains the code, notebooks, generated data, and figures for the paper on the driven quantum-star primitive and the `1/d` L0/G1 hierarchy.

The repository is organized so that the paper figures can be regenerated either from notebooks or directly from Python scripts.

## Directory layout

```text
.
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── star_qdp.py
├── experiments/
│   ├── __init__.py
│   ├── run_main_figures.py
│   ├── run_driven_star_validation.py
│   └── run_uniform_star_LL_to_quantum.py
├── notebooks/
│   ├── 01_generate_main_paper_figures.ipynb
│   ├── 02_driven_star_validation.ipynb
│   └── 03_uniform_star_LL_to_quantum_transition.ipynb
└── figures/
    ├── *.pdf
    ├── *.png
    └── *_data.npz
```

## Notebooks

The notebooks are intended to be run from the `notebooks/` directory. Each notebook detects the repository root automatically and writes figures into `figures/`.

1. `notebooks/01_generate_main_paper_figures.ipynb`

   Regenerates the schematic and static-validation figures:

   - `fig_star_primitive.{pdf,png}`
   - `fig_theory_schematic.{pdf,png}`
   - `fig_aligned_benchmark.{pdf,png}` and `fig_aligned_benchmark_data.npz`
   - `fig_one_over_d_scaling.{pdf,png}` and `fig_one_over_d_scaling_data.npz`
   - `fig_static_inhom_validation.{pdf,png}` and `fig_static_inhom_validation_data.npz`

2. `notebooks/02_driven_star_validation.ipynb`

   Regenerates the fully time-dependent driven-star validation figures:

   - `fig_driven_validation_error.{pdf,png}` and `fig_driven_validation_error_data.npz`
   - `fig_driven_validation_scaling.{pdf,png}` and `fig_driven_validation_scaling_data.npz`

3. `notebooks/03_uniform_star_LL_to_quantum_transition.ipynb`

   Regenerates the homogeneous two-population uniform-star transition figure:

   - `fig_uniform_star_LL_to_quantum.{pdf,png}`
   - `fig_uniform_star_LL_to_quantum_data.npz`

## Python modules and scripts

### `src/star_qdp.py`

Core numerical routines for:

- spin-1/2 coherent states;
- exact aligned homogeneous-star oracle;
- Schur-block homogeneous-star oracle;
- exact sparse/Krylov driven-star oracle;
- L0 and L0+G1 approximations;
- branch-continuous logarithmic errors;
- helper routines for driven-star instances and scaling fits.

### `experiments/run_main_figures.py`

Generates the main schematic and static-validation figures used in the paper. Run with:

```bash
python experiments/run_main_figures.py
```

### `experiments/run_driven_star_validation.py`

Generates the time-dependent driven-star validation figures. Run with:

```bash
python experiments/run_driven_star_validation.py
```

### `experiments/run_uniform_star_LL_to_quantum.py`

Generates the uniform-star LL-to-quantum transition figure. Run with:

```bash
python experiments/run_uniform_star_LL_to_quantum.py
```

## Regenerating all figures

From the repository root, run:

```bash
python experiments/run_main_figures.py
python experiments/run_driven_star_validation.py
python experiments/run_uniform_star_LL_to_quantum.py
```

All generated outputs are written to `figures/`.

## Requirements

The code uses standard scientific Python packages:

```bash
pip install -r requirements.txt
```

The main dependencies are `numpy`, `scipy`, `matplotlib`, `jupyter`, and `nbformat`.

## Notes on reproducibility

- Randomized validation examples use fixed seeds.
- Figure scripts save both publication-style PDFs and PNG previews.
- Numerical data underlying the validation figures are saved as `.npz` files in `figures/`.
- The notebooks are thin wrappers around the Python scripts, so the command-line and notebook workflows produce the same outputs.
