"""Run the research Monte Carlo experiment in headless mode."""
import config
from simulation.monte_carlo import run_monte_carlo_simulations

if __name__ == "__main__":
    config.RUN_MODE = "headless"
    config.validate_configuration()
    run_monte_carlo_simulations()
