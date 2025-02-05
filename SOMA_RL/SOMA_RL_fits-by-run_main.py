
from helpers.plotting import plot_fits_by_run_number

if __name__ == "__main__":

    fit_path = 'SOMA_RL/model results/Standard Models + Novel for 10 Runs/full_fit_data.pkl'
    plot_fits_by_run_number(fit_path)