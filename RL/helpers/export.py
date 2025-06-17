import os
import shutil

class Export:

    # Functions
    def export_fits(self, path: str) -> None:

        """
        Exports the fit results to a specified path.

        Parameters
        ----------
        path : str
            The directory where the fit results will be exported.
        """

        files_to_move = [
            'RL/fits/fit_data.pkl',
            'RL/fits/full_fit_data.pkl',
            'RL/fits/group_AIC_percentages.csv',
            'RL/fits/group_BIC_percentages.csv',
            'RL/fits/modelsimulation_accuracy_data.csv',
            'RL/fits/modelsimulation_choice_data.csv',
            'RL/fits/parameter_outlier_results.pkl',
            'RL/fits/group_AIC.csv',
            'RL/fits/group_BIC.csv',

            'RL/plots/fit-by-runs.png',
            'RL/plots/acutepain_model_simulations.png',
            'RL/plots/chronicpain_model_simulations.png',
            'RL/plots/nopain_model_simulations.png',
            'RL/plots/model_fits_distributions.png',
            'RL/plots/AIC_model_comparisons.png',
            'RL/plots/BIC_model_comparisons.png',
            'RL/plots/parameter_of_interest_comparisons.png',
            
            'RL/stats/pain_fits_linear_results.csv',
            'RL/stats/pain_fits_ttest_results.csv',
            'RL/stats/pain_fits_posthoc_results.csv',
            'RL/stats/param_fit_descriptives.csv',
        ]

        files_to_rename = [
            [f'{path}/fit_data.pkl',       f'{path}/fit_data_FIT.pkl'],
            [f'{path}/full_fit_data.pkl',  f'{path}/full_fit_data_FIT.pkl'],
        ]

        folders_to_move = [
            ['RL/plots/model_behaviours/', f'{path}/model_behaviours'],
            ['RL/plots/fits/',             f'{path}/parameter_fits'],
        ]

        # Conduct the file operations
        if not os.path.exists(path):
            os.makedirs(path)

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

    def export_recovery(self, path: str) -> None:
        
        """
        Exports the model recovery results to a specified path.
        
        Parameters
        ----------
        path : str
            The directory where the model recovery results will be exported.
        """

        files_to_move = [
            'RL/plots/model_recovery.png',
            'RL/fits/fit_data_PARAMETER.pkl',
            'RL/fits/full_fit_data_PARAMETER.pkl',
            'RL/fits/fit_data_MODEL.pkl',
            'RL/fits/full_fit_data_MODEL.pkl',
        ]

        folders_to_move = [
            ['RL/plots/correlations',      f'{path}/correlations'],
            ['RL/data/generated',          f'{path}/data/generated'],
        ]

        # Conduct the file operations
        if not os.path.exists(path):
            os.makedirs(path)

        for file_to_move in files_to_move:
            if os.path.exists(file_to_move):
                shutil.copy(file_to_move, path)

        for folder_to_move in folders_to_move:
            if os.path.exists(folder_to_move[0]):
                if os.path.exists(folder_to_move[1]):
                    shutil.rmtree(folder_to_move[1])
                shutil.copytree(folder_to_move[0], folder_to_move[1])