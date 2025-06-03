import os
import shutil

class Export:

    # Functions
    def export_fits(self, path):

        files_to_move = [
            'SOMA_RL/fits/fit_data.pkl',
            'SOMA_RL/fits/full_fit_data.pkl',
            'SOMA_RL/fits/group_AIC_percentages.csv',
            'SOMA_RL/fits/group_BIC_percentages.csv',
            'SOMA_RL/fits/modelsimulation_accuracy_data.csv',
            'SOMA_RL/fits/modelsimulation_choice_data.csv',
            'SOMA_RL/fits/parameter_outlier_results.pkl',
            'SOMA_RL/fits/group_AIC.csv',
            'SOMA_RL/fits/group_BIC.csv',

            'SOMA_RL/plots/fit-by-runs.png',
            'SOMA_RL/plots/acutepain_model_simulations.png',
            'SOMA_RL/plots/chronicpain_model_simulations.png',
            'SOMA_RL/plots/nopain_model_simulations.png',
            'SOMA_RL/plots/model_fits_distributions.png',
            'SOMA_RL/plots/AIC_model_comparisons.png',
            'SOMA_RL/plots/BIC_model_comparisons.png',
            'SOMA_RL/plots/parameter_of_interest_comparisons.png',
            
            'SOMA_RL/stats/pain_fits_linear_results.csv',
            'SOMA_RL/stats/pain_fits_ttest_results.csv',
            'SOMA_RL/stats/pain_fits_posthoc_results.csv',
            'SOMA_RL/stats/param_fit_descriptives.csv',
        ]

        files_to_rename = [
            [f'{path}/fit_data.pkl',       f'{path}/fit_data_FIT.pkl'],
            [f'{path}/full_fit_data.pkl',  f'{path}/full_fit_data_FIT.pkl'],
        ]

        folders_to_move = [
            ['SOMA_RL/plots/model_behaviours/', f'{path}/model_behaviours'],
            ['SOMA_RL/plots/fits/',             f'{path}/parameter_fits'],
        ]

        # Conduct the file operations
        for file_to_move in files_to_move:
            if os.path.exists(file_to_move):
                shutil.copy(file_to_move, path)

        for old_name, new_name in files_to_rename:
            if os.path.exists(old_name):
                if os.path.exists(new_name):
                    os.remove(new_name)
                os.rename(old_name, new_name)

        for folder_to_move in folders_to_move:
            if os.path.exists(folder_to_move[0]):
                if os.path.exists(folder_to_move[1]):
                    shutil.rmtree(folder_to_move[1])
                shutil.copytree(folder_to_move[0], folder_to_move[1])

    def export_recovery(self, path):

        files_to_move = [
            'SOMA_RL/plots/model_recovery.png',
            'SOMA_RL/fits/fit_data_PARAMETER.pkl',
            'SOMA_RL/fits/full_fit_data_PARAMETER.pkl',
            'SOMA_RL/fits/fit_data_MODEL.pkl',
            'SOMA_RL/fits/full_fit_data_MODEL.pkl',
        ]

        folders_to_move = [
            ['SOMA_RL/plots/correlations',      f'{path}/correlations'],
            ['SOMA_RL/data/generated',          f'{path}/data/generated'],
        ]

        # Conduct the file operations
        for file_to_move in files_to_move:
            if os.path.exists(file_to_move):
                shutil.copy(file_to_move, path)

        for folder_to_move in folders_to_move:
            if os.path.exists(folder_to_move[0]):
                if os.path.exists(folder_to_move[1]):
                    shutil.rmtree(folder_to_move[1])
                shutil.copytree(folder_to_move[0], folder_to_move[1])