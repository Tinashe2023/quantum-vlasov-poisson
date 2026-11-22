# Quantum Vlasov–Poisson

Research code for studying classical discretizations of the one-dimensional, one-velocity (1D1V) Vlasov–Poisson system and investigating its structure for quantum linear-solver approaches.

The repository currently contains a working classical solver and Landau-damping experiment, a set of exploratory notebooks, and saved diagnostics from matrix and quantum-solver studies. The `src/qvp/quantum` and several `src/qvp/analysis` modules are currently empty stubs; the saved diagnostics and notebook material should therefore be read as research artifacts, not as a finished or hardware-executable quantum solver.

## Model and numerical method

The classical code evolves a distribution `f(x, v, t)` on a periodic spatial grid and a bounded velocity grid. It computes the electric field from Poisson's equation using a Fourier-space solve in `x`, evaluates the spatial derivative spectrally and the velocity derivative with centered finite differences (one-sided at the boundaries), then advances the coupled system with fourth-order Runge–Kutta. Maxwellian equilibria and spatially perturbed Maxwellians are provided as initial conditions.

The implementation is a research prototype. In particular, velocity-boundary treatment, time-step and grid convergence, and conservation should be checked for each use case; the README does not imply validated production accuracy.

## Repository layout

```text
src/qvp/
  classical/       Grid, Poisson solve, Vlasov right-hand side, RK4 step
  physics/         Maxwellian initial distributions
  analysis/        Observable calculations (other modules are stubs)
  quantum/         Reserved for quantum encodings and solver implementations
experiments/
  landau_damping/  Landau-damping run and convergence-rate estimation
notebooks/         Stepwise numerical, stability, matrix, and quantum studies
results/           Saved diagnostics and comparison outputs
tests/             Unit tests for grid, Poisson, and operator behavior
data/              Input data area
paper/             Research manuscript area
```

Notebook outputs and files under `results/` are generated or saved research artifacts. They may reflect a particular run and are not automatically refreshed when source code changes.

## Setup

Use Python 3.10 or newer. From the repository root, create and activate a virtual environment, then install the packages used by the current code and notebooks:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib jupyter ipykernel pytest
```

`environment.yml` is currently empty, so it does not yet provide a Conda environment specification. The package uses a `src/` layout and has no packaging metadata at present. To import `qvp` in a shell or when running an experiment, set the source directory on `PYTHONPATH`:

```bash
# macOS/Linux
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# Windows PowerShell
$env:PYTHONPATH = "$PWD/src;" + $env:PYTHONPATH
```

The Landau-damping experiment also adds `src/` to the import path itself, so it can be run directly from the repository root:

```bash
python experiments/landau_damping/convergence.py
```

Open notebooks from the repository root so their relative paths resolve:

```bash
jupyter lab
```

## Quick start: classical solver

This example creates a perturbed Maxwellian, takes one RK4 step, and computes the resulting electric field:

```python
from qvp.classical.grid import create_grid
from qvp.classical.poisson import calculate_electric_field
from qvp.classical.solver import rk4_step
from qvp.physics.distributions import perturbed_maxwellian

L, Nx, Nv = 4 * 3.141592653589793, 64, 96
x, v, X, V, dx, dv, kx = create_grid(
    L=L, v_min=-6.0, v_max=6.0, Nx=Nx, Nv=Nv
)
f = perturbed_maxwellian(V, X, alpha=0.01, k=0.5)
f = rk4_step(f, dt=0.01, V=V, kx=kx, dv=dv)
E, rho, phi_hat = calculate_electric_field(f, dv, kx)
```

`create_grid` returns `(x, v, X, V, dx, dv, kx)`. The distribution is stored with shape `(Nx, Nv)`.

## Notebooks and experiment

The notebooks are numbered by topic: phase-space grid setup, classical evolution, Landau damping, two-stream instability, convergence, matrix analysis, quantum formulation, and resource benchmarking. There are two notebooks numbered `05`; open the one matching the topic in its filename.

The Landau-damping driver in `experiments/landau_damping/convergence.py` returns time histories for field amplitude and conserved quantities, along with a damping-rate estimate obtained by fitting the logarithm of detected electric-field peaks. The estimate is sensitive to the chosen grid, time step, fit interval, and peak selection; inspect those choices before interpreting it.

## Tests

Run the current unit tests from the repository root with `src/` on the import path:

```bash
python -m pytest
```

## Research status and reproducibility

Quantum solver names in the notebook and results collection include HHL, VQLS, QSVT, and structured-LCU investigations. The corresponding Python modules under `src/qvp/quantum/` are placeholders at present, so these saved comparisons are not callable solver implementations in this package. Record the Python and dependency versions, grid parameters, initial conditions, time step, and analysis choices when reproducing numerical results. Generated caches and many raw/processed data artifacts are excluded by `.gitignore`.
