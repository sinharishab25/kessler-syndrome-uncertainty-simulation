# Positional Tracking Uncertainty and Observed Cascade Onset in a Simplified LEO Simulation

**Research by Rishab Sinha**  
**Project:** *The Influence of Positional Tracking Uncertainty on Observed Cascade Onset in a Simplified LEO Simulation*

## Overview

This repository contains the Python implementation of a Monte Carlo model investigating how positional tracking uncertainty affects the **detection and apparent timing** of cascading collisions in a simplified Low-Earth Orbit (LEO) environment.

The simulation maintains two distinct histories:

- **Physical (true) layer:** object motion, actual collisions, fragmentation and collision ancestry.
- **Observation layer:** collision and cascade classifications calculated from positions perturbed by independent, zero-mean Gaussian tracking errors.

Observation noise does not alter the physical layer. A cascade onset is defined operationally as the first time a collision ancestry reaches **generation 3**. This is a research-specific threshold, **not** a universal physical definition of Kessler Syndrome.

## Model and methods

- Two-dimensional periodic observation window: **1,000 × 600 km**.
- Constant-velocity motion between collisions; no gravity, drag, orbital maneuvers or full orbital dynamics.
- Continuous closest-approach checks within discrete simulation steps, with periodic minimum-image displacement.
- Simplified size-dependent fragmentation and ancestry tracking.
- Isotropic Gaussian position error with coordinate standard deviation **σ**.
- Separate true and observed cascade-onset times, with onset bias **B = T_observed − T_true**.
- Deterministic seeds allow comparisons across uncertainty settings without changing the true collision histories.

The paper's experiment used **N₀ = 6 and 10**, **σ = 0, 1, 2 and 4 km**, and **100 repetitions per condition** (800 runs total), over a maximum of 1,000 *model years*. Model years are internal simulation time and do not represent real-world predictions.

## Installation

Install Python 3 and run the following commands from the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Running the experiment

```bash
python run.py
```

`run.py` starts the **headless Monte Carlo** workflow, which is the workflow relevant to the research paper. Parameters are configured in `config.py`. The supplied configuration contains the 6/10 population sweep, 0/1/2/4 km uncertainty sweep and 100 repetitions per condition. The full batch may take substantial time.

For a quicker preliminary run, reduce `MONTE_CARLO_RUN_COUNT` in `config.py`, and optionally shorten `MODEL_DURATION`. Such changes do **not** reproduce the paper's full experiment.

Generated results and figures are written to `outputs/` when the program runs. **No previously generated results or output files are included in this repository.** The directory is excluded from Git using `.gitignore`.

**Time-step clarification:** `config.py` sets `MODEL_DT = 0.01` model years and `SIMULATION_SPEED_MULTIPLIER = 10`. The effective simulation time step is therefore `0.01 × 10 = 0.1` model years, matching Appendix Table A1 of the research paper. **Do not change `MODEL_DT` to `0.1`** to match the paper; that would make the effective step `1.0` model years with the current multiplier. The recorded experiment used an effective step of `0.1` model years. The full reported numerical results have not been independently reproduced by this repository preparation.

## Repository contents

- `run.py`: simple headless research entry point.
- `config.py`: model, uncertainty, fragmentation and experiment parameters.
- `simulation/`: collision detection, physical and observation histories, fragmentation, Monte Carlo execution and required supporting modules.
- `analysis/`: analysis and graph-generation modules called by the batch workflow.
- `requirements.txt`: third-party Python dependencies.
- `.gitignore`: prevents committing generated outputs and local environments.

Development backups, virtual environments, previous result datasets, screenshots, test artifacts and visual-only or benchmarking entry points have been omitted. The research execution path is retained rather than rewritten into a different simulation.

## Interpretation and limitations

The research found that increasing positional uncertainty could strongly inflate *observed* generation-3 cascade classifications without changing true physical collision histories. Of 303 cascade-classification errors in the reported 800-run experiment, 297 were false-positive cascade detections and six were missed cascades. Where both histories reached onset, observed onset was generally earlier.

These findings apply to this simplified computational model. The artificial geometry, effective object radii, time scaling and fragmentation assumptions mean that the results should **not** be interpreted as predictions for actual LEO debris populations or operational conjunction assessments.

## AI assistance and attribution

The Python implementation was generated with AI assistance. The research involved specifying and developing the research question and experimental methodology, directing the simulation work, checking model behavior, and interpreting the results with research mentorship. AI assistance should not be confused with independently hand-written implementation. Readers should evaluate the documented methods, source code and reproducibility independently.

## Research paper

*The Influence of Positional Tracking Uncertainty on Observed Cascade Onset in a Simplified LEO Simulation*, Rishab Sinha (2026). The complete paper is not bundled with this code-only repository.
