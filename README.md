# Positional Tracking Uncertainty and Observed Cascade Onset in a Simplified LEO Simulation

**Research by Rishab Sinha**

**Paper:** *The Influence of Positional Tracking Uncertainty on Observed Cascade Onset in a Simplified LEO Simulation* (2026)

## Overview

This repository contains the Python implementation of a Monte Carlo model investigating how positional tracking uncertainty affects the detection and apparent timing of cascading collisions in a simplified Low Earth Orbit (LEO) environment.

The model maintains two separate layers:

- **Physical layer:** Tracks actual collisions and fragmentation, including debris ancestry.
- **Observation layer:** Uses positions affected by Gaussian tracking errors to identify apparent collisions and cascades.

Tracking errors affect only the observation layer. They do not change physical collisions or object motion.

Cascade onset is defined as the first instance when collision ancestry reaches generation 3. This threshold is specific to the study and is not a universal definition of Kessler Syndrome.

## Model and Methods

- **Simulation space:** Two-dimensional periodic window measuring 1,000 × 600 km.
- **Motion:** Constant velocity between collisions, without gravity or orbital dynamics.
- **Collision detection:** Continuous closest-approach calculations within each simulation step.
- **Fragmentation:** Simplified size-dependent rules with debris ancestry tracking.
- **Tracking uncertainty:** Independent Gaussian position errors with coordinate standard deviation σ.
- **Cascade onset:** First occurrence of generation 3 or higher.
- **Onset bias:** B = T_observed − T_true.

The experiment tested two initial populations (N₀ = 6 and 10) at four uncertainty levels (σ = 0, 1, 2 and 4 km). Each condition was repeated 100 times, giving **800 simulation runs**.

The maximum simulation duration was 1,000 model years. These represent internal simulation time rather than actual years in orbit.

The same random seeds were paired across uncertainty levels to keep the physical collision histories consistent.

## Key Findings

Increasing tracking uncertainty caused the model to report more cascades, even though the true cascade probabilities remained unchanged.

At σ = 4 km:

- **N₀ = 6:** Observed cascade probability was 68%, compared with a true probability of 10%.
- **N₀ = 10:** Observed cascade probability was 100%, compared with a true probability of 49%.

Across all 800 runs, there were 303 cascade classification errors. Of these, **297 were false detections** and six were missed cascades.

When both layers detected a cascade, the observed onset generally occurred earlier than the true onset.

These findings suggest that positional uncertainty can affect the reliability of cascade detection within the simplified model.

## Installation

Install Python 3 and the required libraries from the repository directory:

```bash
python -m pip install -r requirements.txt
```

## Running the Simulation

Run the experiment using:

```bash
python run.py
```

This starts the headless Monte Carlo workflow used for the research.

The experimental settings are stored in `config.py`. The supplied configuration uses the same initial populations and uncertainty levels as the paper, with 100 repetitions per condition.

The full experiment may take considerable time.

For a shorter test, reduce `MONTE_CARLO_RUN_COUNT` in `config.py`. The simulation duration can also be reduced through `MODEL_DURATION`. These changes will not reproduce the complete experiment.

Generated results and figures are saved in `outputs/`. Previously generated results are not included in the repository.

**Time-step note:** The effective simulation step is 0.1 model years. This comes from `MODEL_DT = 0.01` and `SIMULATION_SPEED_MULTIPLIER = 10` in `config.py`. These settings match the time step reported in Appendix Table A1 of the paper. Changing `MODEL_DT` to 0.1 without adjusting the multiplier would increase the effective step to 1.0 model years.

The complete reported numerical results have not been independently reproduced from this repository.

## Repository Contents

- `run.py`: Starts the headless research simulation.
- `config.py`: Contains the model parameters and experimental settings.
- `simulation/`: Contains the code for motion and collision detection. It also handles fragmentation, tracking uncertainty and Monte Carlo execution.
- `analysis/`: Contains the code for analysing results and generating figures.
- `requirements.txt`: Lists the required Python libraries.
- `.gitignore`: Excludes generated outputs and local files.
- `research_paper.pdf`: Contains the complete research paper.

Development backups and temporary files have been excluded. The original research execution workflow has been retained.

## Interpretation and Limitations

This is a simplified computational model rather than a realistic simulation of the current LEO environment.

The model uses a two-dimensional periodic space and enlarged collision radii. It assumes constant-velocity motion between collisions and simplified fragmentation rules. Gravity and atmospheric drag are not included.

The results demonstrate how positional uncertainty can affect cascade detection under these assumptions. They should not be used to predict actual orbital debris events or assess operational collision risks.

## AI Assistance and Attribution

AI tools assisted with generating and debugging the Python implementation. The research question and experimental methodology were developed with guidance from a research mentor. The simulation work was directed and its behaviour was checked by the researcher, who also interpreted the results.

The implementation should not be treated as entirely handwritten code. The methods and limitations are documented in the accompanying research paper.

## Research Paper

**The Influence of Positional Tracking Uncertainty on Observed Cascade Onset in a Simplified LEO Simulation**

Rishab Sinha (2026)

[Read the full research paper](research_paper.pdf)
