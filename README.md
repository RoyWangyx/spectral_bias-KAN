# On the Expressiveness and Spectral Bias of KANs

This repository contains the code used to generate the main numerical experiments for the paper **“On the expressiveness and spectral bias of KANs”** by Yixuan Wang, Jonathan W. Siegel, Ziming Liu, and Thomas Y. Hou (ICLR 2025). The paper studies two main themes: (i) theoretical comparisons between KANs and MLPs in terms of representation/approximation, and (ii) empirical comparisons of their spectral bias on several regression and PDE tasks. In particular, the paper reports experiments on **1D wave fitting**, **Gaussian random field regression**, and **High frequency Poisson equation solving**, and uses them to show that KANs are typically less biased toward low frequencies than MLPs. 

## What is in this archive


```text
code/
├── 1D_frequency/     # 1D multi-frequency wave fitting (Figures 1–2 style experiments)
├── GRF/              # Gaussian random field regression (Figures 3–4 style experiments)
├── PDE/              # PDE experiments, including additional higher-dimensional scripts
├── PDE_drm_1d/       # 1D Deep Ritz Method Poisson experiments
├── image/            # extra image-fitting / compression-style experiments
└── README.md
```

The three folders most directly tied to the paper’s main experimental narrative are:

- `1D_frequency/`: 1D wave fitting for spectral-bias visualization.
- `GRF/`: regression on samples from Gaussian random fields.
- `PDE_drm_1d/` and parts of `PDE/`: Poisson / Deep Ritz experiments.

## Main paper experiments represented here

### 1) 1D waves of different frequencies

The paper fits linear combinations of sinusoidal modes with either equal amplitudes or increasing amplitudes, compares Fourier-mode learning dynamics between MLPs and KANs, and reports that KANs learn high frequencies much earlier than standard MLPs. In the paper, MLPs are trained for longer than KANs in this benchmark, and the resulting plots correspond to Figures 1 and 2. 

Relevant files:

- `1D_frequency/1D_frequency.py` — KAN runs
- `1D_frequency/1D_frequency_mlp.py` — MLP runs
- `1D_frequency/gpu.sh` / `gpumlp.sh` — SLURM array launchers

What these scripts do:

- generate a 1D signal on `[0,1]` from a sum of sines,
- train either a KAN or an MLP on sampled points,
- record predictions during training,
- compute FFT magnitudes over time,
- save heatmap figures as `.png` files in the same folder.

### 2) Gaussian random field regression

The paper next studies regression on functions sampled from a Gaussian random field, with multiple input dimensions and covariance scales. It reports training/test loss comparisons and discusses overfitting behavior, including the effect of increasing sample size. These correspond to Figures 3 and 4.

Relevant files:

- `GRF/GRF.py` — baseline MLP regression
- `GRF/GRF_KAN.py` — KAN regression with grid refinement
- `GRF/GRF_overfit.py` — overfitting test for MLP
- `GRF/GRF_KAN_overfit.py` — overfitting test for KAN
- `GRF/GRF_reluk.py` / `GRF_overfit_reluk.py` — ReLU^k variants
- `GRF/*.sh` — SLURM launchers
- `GRF/plot.ipynb` — plotting notebook

What these scripts do:

- sample a GRF approximately via a KL expansion,
- split into train/test sets,
- fit MLP or KAN models,
- save training and test losses as `.txt` files,
- optionally study undersampling / overfitting behavior.

### 3) Poisson / Deep Ritz PDE experiments

The paper’s PDE benchmark is a **1D Poisson equation with high-frequency solutions** solved in variational form using the **Deep Ritz Method**. It compares relative `L^2` and `H^1` errors between MLPs and KANs across frequencies, showing that KANs remain much more stable as frequency increases; this is Figure 5 in the paper.

Relevant files:

- `PDE_drm_1d/pde_combined.py` — compact 1D Deep Ritz benchmark comparing KAN vs MLP
- `PDE_drm_1d/gpukantest.sh` — SLURM launcher
- `PDE/` — additional PDE scripts, including alternative setups and higher-dimensional variants
- `PDE/PDE_drm_2d/pde2d.py` — 2D Deep Ritz-style experiment

What these scripts do:

- solve a Poisson equation with Dirichlet boundary conditions,
- evaluate both `L^2` and `H^1` errors against the known exact solution,
- compare standard MLPs to KANs with grid refinement,
- save error curves as `.txt` files and figures as `.png` files.

## Dependencies

Core Python packages:

- `python>=3.9`
- `torch`
- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `Pillow`
- `jupyter` (for the notebooks)
- `pykan`


## Important implementation notes

- Most scripts set `torch.set_default_dtype(torch.float64)`. Running in double precision is part of the current setup.
- Many scripts assume that you run them **from inside their own folder**, since outputs are saved with relative paths.
- Several KAN scripts create or use a local `model/` directory because `pykan` can auto-save checkpoints/history.
- The provided shell scripts are written for a **SLURM cluster** and contain machine-specific email/job settings. You will likely want to edit those before using them.
- Almost all experiment drivers take two positional arguments:

```bash
python script.py <task_id> <num_tasks>
```

This is designed for job arrays. For a local run, using `0 1` is the easiest way to execute the full parameter sweep in one process.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{wang2024expressiveness,
  title   = {On the expressiveness and spectral bias of KANs},
  author  = {Wang, Yixuan and Siegel, Jonathan W. and Liu, Ziming and Hou, Thomas Y.},
  journal = {arXiv preprint arXiv:2410.01803},
  year    = {2024}
}
```

Paper link: arXiv:2410.01803, later listed on the arXiv entry as **ICLR 2025**.

