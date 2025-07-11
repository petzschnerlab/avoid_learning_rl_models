import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy import stats

from models.rl_models import RLModel

if 'Helvetica' in set(f.name for f in font_manager.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['font.size'] = 18

class Plotting:

    def plot_simulations(self,
                         accuracy: dict,
                            prediction_errors: dict,
                            values: dict,
                            choice_rates: dict,
                            models: list,
                            group: str,
                            dataloader: object = None,
                            alpha: float = 0.75) -> None:
        
        """
        Plots the model simulations for a given group.
        
        Parameters
        ----------
        accuracy : dict
            Dictionary containing accuracy data for each model.
        prediction_errors : dict
            Dictionary containing prediction error data for each model.
        values : dict
            Dictionary containing value data for each model.
        choice_rates : dict
            Dictionary containing choice rate data for each model.
        models : list
            List of model names to plot.
        group : str
            The group for which to plot the simulations (e.g., 'no pain', 'acute pain', 'chronic pain').
        dataloader : DataLoader, optional
            An instance of DataLoader to fetch empirical data for comparison. Default is None.
        alpha : float, optional
            Transparency level for the plots. Default is 0.75.

        Returns
        -------
        None
        """
        
        if dataloader is not None:

            #Get empirical learning curves 
            empirical_data = copy.copy(dataloader.get_data()[0])
            empirical_data = empirical_data[empirical_data['pain_group'] == group]
            emp_accuracy = empirical_data.groupby(['context_val_name','trial_number'], observed=False)['accuracy'].mean().reset_index()
            emp_accuracy['context_val_name'] = emp_accuracy['context_val_name'].replace({'Loss Avoid': 'Punish'})

            #Get empirical choice rates
            empirical_data = copy.copy(dataloader.get_data()[1])
            empirical_data = empirical_data[empirical_data['pain_group'] == group]
            empirical_data['stim_id'] = empirical_data['state'].apply(lambda x: x.split(' ')[1])
            emp_choice_rate = {}
            for stimulus in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']:
                stimulus_data = empirical_data.loc[empirical_data['stim_id'].apply(lambda x: stimulus in x)].copy()
                stimulus_data.loc[:,'stim_index'] = stimulus_data.loc[:,'stim_id'].apply(lambda x: 0 if stimulus == x[0] else 1)
                stimulus_data['stim_chosen'] = stimulus_data.apply(lambda x: int(x['action'] == x['stim_index']), axis=1)
                emp_choice_rate[stimulus] = int((stimulus_data['stim_chosen'].sum()/len(stimulus_data))*100)

            pairs = [['A','C'],['B','D'],['E','G'],['F','H']]
            emp_choice_rates = {}
            for pair in pairs:
                emp_choice_rates[pair[0]] = (emp_choice_rate[pair[0]] + emp_choice_rate[pair[1]])/2
            emp_choice_rates['N'] = emp_choice_rate['N']

        colors = self.get_colors('condition')
        bi_colors = self.get_colors('condition_2')
        val_colors = self.get_colors('condition')[:-1]
        fig, ax = plt.subplots(4, np.max((2,len(models))), figsize=(4*len(models), 15))
        for i, m in enumerate(models):

            number_of_participants = accuracy[m]['run'].nunique()

            #Plot accuracy
            model_accuracy = accuracy[m].groupby(['context','trial_total','run'], observed=False).mean().reset_index()
            model_accuracy['context'] = pd.Categorical(model_accuracy['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
            for ci, context in enumerate(['Reward', 'Loss Avoid']):
                CIs = model_accuracy.groupby(['context','trial_total'], observed=False)['accuracy'].sem()*stats.t.ppf(0.975, number_of_participants-1)
                averaged_accuracy = model_accuracy.groupby(['context','trial_total'], observed=False).mean().reset_index()
                context_accuracy = averaged_accuracy[averaged_accuracy['context'] == context]['accuracy'].reset_index(drop=True).astype(float)*100
                context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)*100
                ax[0, i].fill_between(context_accuracy.index, context_accuracy - context_CIs, context_accuracy + context_CIs, alpha=0.1, color=bi_colors[ci], edgecolor='none')
                ax[0, i].plot(context_accuracy, color=bi_colors[ci], alpha = alpha, label=context.replace('Loss Avoid', 'Punish'))
                if dataloader is not None:
                    ax[0, i].plot(emp_accuracy[emp_accuracy['context_val_name'] == context]['trial_number'], emp_accuracy[emp_accuracy['context_val_name'] == context]['accuracy'], color=bi_colors[ci], linestyle='dashed', alpha= alpha)

            ax[0, i].set_title(m)
            ax[0, i].set_ylim([25, 100])
            if i == 0:
                ax[0, i].set_ylabel('Accuracy (%)')
            ax[0, i].set_xlabel('Trial')
            if i == len(models)-1:
                ax[0, i].legend(loc='lower right', frameon=False)
            ax[0, i].spines['top'].set_visible(False)
            ax[0, i].spines['right'].set_visible(False)

            #Plot prediction errors
            model_pe = prediction_errors[m].groupby(['context','trial_total', 'run'], observed=False).mean().reset_index()
            model_pe['context'] = pd.Categorical(model_pe['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
            for ci, context in enumerate(['Reward', 'Loss Avoid']):
                CIs = model_pe.groupby(['context','trial_total'], observed=False)['averaged_pe'].sem()*stats.t.ppf(0.975, number_of_participants-1)
                averaged_pe = model_pe.groupby(['context','trial_total'], observed=False).mean().reset_index()
                context_pe = averaged_pe[averaged_pe['context'] == context]['averaged_pe'].reset_index(drop=True)
                context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)
                ax[1, i].fill_between(context_pe.index, context_pe - context_CIs, context_pe + context_CIs, alpha=0.1, color=bi_colors[ci], edgecolor='none')
                ax[1, i].plot(context_pe, color=bi_colors[ci], alpha = alpha, label=context.replace('Loss Avoid', 'Punish'))
            ax[1, i].set_ylim([-.75, .75])
            if i == 0:
                ax[1, i].set_ylabel('Prediction Error')
            ax[1, i].set_xlabel('Trial')
            if i == len(models)-1:
                ax[1, i].legend(loc='lower right', frameon=False)
            ax[1, i].spines['top'].set_visible(False)
            ax[1, i].spines['right'].set_visible(False)
            ax[1, i].axhline(0, linestyle='--', color='grey', alpha=.5)

            #Plot values
            model_values = values[m].groupby(['context','trial_total', 'run'], observed=False).mean().reset_index()
            model_values['context'] = pd.Categorical(model_values['context'], categories=['Reward', 'Loss Avoid'], ordered=True)
            for ci, context in enumerate(['Reward', 'Loss Avoid']):
                for vi, val in enumerate(['values1', 'values2']):
                    CIs = model_values.groupby(['context','trial_total'], observed=False)[val].sem()*stats.t.ppf(0.975, number_of_participants-1)
                    averaged_values = model_values.groupby(['context','trial_total'], observed=False).mean().reset_index()
                    context_values = averaged_values[averaged_values['context'] == context][val].reset_index(drop=True)
                    context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True)
                    ax[2, i].fill_between(context_values.index, context_values - context_CIs, context_values + context_CIs, alpha=0.1, color=val_colors[ci*2+vi], edgecolor='none')
                    ax[2, i].plot(context_values, color=val_colors[ci*2+vi], alpha = alpha, label=['High Reward', 'Low Reward', 'Low Punish', 'High Punish'][ci*2+vi])
            ax[2, i].set_ylim([-1, 1])
            if i == 0:
                ax[2, i].set_ylabel('q/w/h Value')
            ax[2, i].set_xlabel('Trial')
            if i == len(models)-1:
                ax[2, i].legend(loc='lower left', frameon=False, ncol=2)
            ax[2, i].spines['top'].set_visible(False)
            ax[2, i].spines['right'].set_visible(False)
            ax[2, i].axhline(0, linestyle='--', color='grey', alpha=.5)

            #Plot choice rates        
            ax[3, i].bar(['HR', 'LR', 'LP', 'HP', 'N'], choice_rates[m].mean(axis=0), color=colors, alpha = .5)
            ax[3, i].errorbar(['HR', 'LR', 'LP', 'HP', 'N'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem()* stats.t.ppf(0.975, number_of_participants-1), fmt='.', color='grey')
            if dataloader is not None:
                ax[3, i].scatter(['HR', 'LR', 'LP', 'HP', 'N'], list(emp_choice_rates.values()), color='grey', marker='D', alpha=alpha)
            ax[3, i].set_ylim([0, 100])
            if i == 0:
                ax[3, i].set_ylabel('Choice Rate (%)')
            ax[3, i].spines['top'].set_visible(False)
            ax[3, i].spines['right'].set_visible(False)

        #Metaplot settings
        fig.tight_layout()
        fig.suptitle(f"{group.title()}", x=0.001, y=.999, ha='left', fontsize=16)
        fig.savefig(os.path.join('RL','plots',f"{group.replace(' ','')}_model_simulations.png"))
        fig.savefig(os.path.join('RL','plots',f"{group.replace(' ','')}_model_simulations.svg"), format='svg')
        
        #Close figure
        plt.close()

    def plot_simulations_behaviours(self,
                                    accuracy: dict,
                                    choice_rates: dict,
                                    models: list,
                                    groups: list,
                                    dataloader: object = None,
                                    rolling_mean: int = None,
                                    plot_type: str = 'raincloud',
                                    alpha: float = 0.75,
                                    binned_trial: bool = False,
                                    subplot_title: list[str] = None) -> None:

        """
        Plots the model simulations for a given group, including accuracy, choice rates, and empirical data if available.

        Parameters
        ----------
        accuracy : dict
            Dictionary containing accuracy data for each model and group.
        choice_rates : dict
            Dictionary containing choice rate data for each model and group.
        models : list
            List of model names to plot.
        groups : list
            List of groups to plot (e.g., ['no pain', 'acute pain', 'chronic pain']).
        dataloader : DataLoader, optional   
            An instance of DataLoader to fetch empirical data for comparison. Default is None.
        rolling_mean : int, optional    
            The window size for rolling mean smoothing of the accuracy data. Default is None (no smoothing).
        plot_type : str, optional
            Type of plot to generate for choice rates ('raincloud' or 'bar'). Default is 'raincloud'.
        alpha : float, optional
            Transparency level for the plots. Default is 0.75.
        binned_trial : bool, optional
            If True, uses binned trials for the accuracy plot. Default is False.
        subplot_title : list[str], optional
            List of titles for each subplot. If None, no titles are set. Default is None.
       
        Returns
        -------
        None
        """

        bi_colors = self.get_colors('condition_2')

        if dataloader is not None:

            #Get empirical learning curves 
            empirical_data = copy.copy(dataloader.get_data()[0])
            emp_accuracy = empirical_data.groupby(['context_val_name','trial_number', 'pain_group'], observed=False)['accuracy'].mean().reset_index()
            emp_accuracy_group = {}
            for group in groups:
                group_accuracy = emp_accuracy[emp_accuracy['pain_group'] == group].copy()
                group_accuracy.drop('pain_group', axis=1, inplace=True)
                emp_accuracy_group[group] = group_accuracy
                
            #Get empirical choice rates
            empirical_data = copy.copy(dataloader.get_data()[1])
            empirical_data['stim_id'] = empirical_data['state'].apply(lambda x: x.split(' ')[1])
            emp_choice_groups = {}
            for group in groups:
                emp_choice_rate = {}
                for stimulus in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'N']:
                    stimulus_data = empirical_data.loc[empirical_data['stim_id'].apply(lambda x: stimulus in x)].copy()
                    stimulus_data = stimulus_data[stimulus_data['pain_group'] == group].copy()
                    stimulus_data.loc[:,'stim_index'] = stimulus_data.loc[:,'stim_id'].apply(lambda x: 0 if stimulus == x[0] else 1)
                    stimulus_data['stim_chosen'] = stimulus_data.apply(lambda x: int(x['action'] == x['stim_index']), axis=1)
                    emp_choice_rate[stimulus] = int((stimulus_data['stim_chosen'].sum()/len(stimulus_data))*100)
                pairs = [['A','C'],['B','D'],['E','G'],['F','H']]
                emp_choice_rates = {}
                for pair in pairs:
                    emp_choice_rates[pair[0]] = (emp_choice_rate[pair[0]] + emp_choice_rate[pair[1]])/2
                emp_choice_rates['N'] = emp_choice_rate['N']
                emp_choice_groups[group] = emp_choice_rates
                        
        # Iterate over each model
        for i, m in enumerate(models):
            
            # Create a figure with subplots for each group
            fig, ax = plt.subplots(2, len(groups), figsize=(5 * len(groups), 10))

            for gi, group in enumerate(['no pain', 'acute pain', 'chronic pain']):
                number_of_participants = accuracy[m][group]['run'].nunique()

                # Plot accuracy
                if binned_trial:
                    accuracy[m][group]['binned_trial'] = pd.cut(accuracy[m][group]['trial_total'], bins=[0, 6, 12, 18, 24], labels=['Early', 'Mid-Early', 'Mid-Late', 'Late'], include_lowest=True)

                trial_type = 'binned_trial' if binned_trial else 'trial_total'
                model_accuracy = accuracy[m][group].groupby(['context', trial_type, 'run'], observed=False).mean().reset_index()
                model_accuracy['context'] = pd.Categorical(model_accuracy['context'], categories=['Reward', 'Loss Avoid'], ordered=True)

                for ci, context in enumerate(['Reward', 'Loss Avoid']):
                    CIs = model_accuracy.groupby(['context', trial_type], observed=False)['accuracy'].sem() * stats.t.ppf(0.975, number_of_participants-1)
                    CIs = CIs.reset_index()
                    averaged_accuracy = model_accuracy.groupby(['context', trial_type], observed=False).mean().reset_index()
                    context_accuracy = averaged_accuracy[averaged_accuracy['context'] == context][[trial_type, 'accuracy']].set_index(trial_type)['accuracy']
                    if rolling_mean is not None and binned_trial == False:
                        context_accuracy = context_accuracy.rolling(rolling_mean, min_periods=1, center=True).mean()
                    context_CIs = CIs[CIs['context'] == context].set_index(trial_type)['accuracy']
                    
                    if binned_trial:
                        context_accuracy = context_accuracy.reindex(['Early', 'Mid-Early', 'Mid-Late', 'Late'])
                        context_CIs = context_CIs.reindex(['Early', 'Mid-Early', 'Mid-Late', 'Late'])
                        context_CIs = context_CIs.values
                        ax[0, gi].fill_between(context_accuracy.index, context_accuracy.values - context_CIs, context_accuracy.values + context_CIs, alpha=0.1, color=bi_colors[ci], edgecolor='none')
                        ax[0, gi].plot(context_accuracy.index, context_accuracy, color=bi_colors[ci], label=context.title().replace('Loss Avoid', 'Punish'), linewidth=3, alpha=0.25)
                        ax[0, gi].scatter(context_accuracy.index, context_accuracy, color=bi_colors[ci], s=10, alpha=alpha)
                    else:
                        ax[0, gi].fill_between(context_accuracy.index+1, context_accuracy - context_CIs, context_accuracy + context_CIs, alpha=0.1, color=bi_colors[ci], edgecolor='none')
                        ax[0, gi].plot(context_accuracy.index+1, context_accuracy, color=bi_colors[ci], alpha=alpha, label=context.replace('Loss Avoid', 'Punish'), linewidth=3)

                    if dataloader is not None:
                        if binned_trial:
                            emp_accuracy_group[group]['binned_trial'] = pd.cut(emp_accuracy_group[group]['trial_number'], bins=[0, 6, 12, 18, 24], labels=['Early', 'Mid-Early', 'Mid-Late', 'Late'], include_lowest=True)
                            trials = context_accuracy.index
                            accuracies = emp_accuracy_group[group][emp_accuracy_group[group]['context_val_name'] == context].groupby('binned_trial')['accuracy'].mean()
                            ax[0, gi].scatter(trials, accuracies, color=bi_colors[ci], marker='D', alpha=alpha)
                        else:
                            trials = emp_accuracy_group[group][emp_accuracy_group[group]['context_val_name'] == context]['trial_number']
                            accuracies = emp_accuracy_group[group][emp_accuracy_group[group]['context_val_name'] == context]['accuracy']
                            if rolling_mean is not None:
                                accuracies = accuracies.rolling(rolling_mean, min_periods=1, center=True).mean()
                            ax[0, gi].plot(trials, accuracies, color=bi_colors[ci], linestyle='dashed', alpha=alpha, linewidth=2)
                
                ax[0, gi].set_title(f'{group.title()}')
                ax[0, gi].set_ylim([40, 100])
                ax[0, gi].legend(loc='lower right', frameon=False)
                ax[0, gi].set_ylabel('Accuracy (%)')
                if binned_trial:
                    ax[0, gi].set_xlabel(' ')
                    tick_positions = [0, 1, 2, 3]
                    tick_labels = ['Early', 'Mid-Early', 'Mid-Late', 'Late']
                    ax[0, gi].set_xticks(tick_positions)
                    ax[0, gi].set_xticklabels(tick_labels, rotation=45, ha='right')
                else:
                    ax[0, gi].set_xlabel('Trial')
                ax[0, gi].spines['top'].set_visible(False)
                ax[0, gi].spines['right'].set_visible(False)
                if binned_trial == False:
                    ax[0, gi].set_xticks(np.arange(0, 25, 4))
                if subplot_title and gi == 0:
                    ax[0, gi].annotate(subplot_title[0],
                                    xy=(-.25, 1.15),
                                    xytext=(0, 0),
                                    xycoords='axes fraction',
                                    textcoords='offset points',
                                    ha='left',
                                    va='top',
                                    fontweight='bold')

                # Plot choice rates
                _, t_scores = self.compute_n_and_t(choice_rates[m][group], None)
                choice_data_long = choice_rates[m][group].stack().reset_index()[['level_1', 0]]
                choice_data_long.columns = ['stimulus', 'choice_rate']
                choice_data_long['stimulus'] = choice_data_long['stimulus'].replace({'A': 'High\nReward', 'B': 'Low\nReward', 'E': 'Low\nPunish', 'F': 'High\nPunish', 'N': 'Novel'})
                choice_data_long.set_index('stimulus', inplace=True)

                if plot_type == 'raincloud':
                    self.raincloud_plot(data=choice_data_long, ax=ax[1, gi], t_scores=t_scores)
                elif plot_type == 'bar':
                    self.bar_plot(data=choice_data_long, ax=ax[1, gi], t_scores=t_scores)
                else:
                    raise ValueError(f"Invalid plot type {plot_type}. Choose either 'raincloud' or 'bar'.")
                
                ax[1, gi].errorbar(np.arange(1,choice_data_long.index.nunique()+1), choice_rates[m][group].mean(axis=0), yerr=choice_rates[m][group].sem()*stats.t.ppf(0.975, number_of_participants-1), fmt='.', color='grey')
                
                if dataloader is not None:
                    ax[1, gi].scatter(np.arange(1,choice_data_long.index.nunique()+1), list(emp_choice_groups[group].values()), color='grey', marker='D', alpha=alpha)

                #Set x-ticks and labels for the choice rate plot
                ax[1, gi].set_xticks(np.arange(1,choice_data_long.index.nunique()+1), ['HR', 'LR', 'LP', 'HP', 'N'])
                ylims = [-4, 104] if plot_type == 'raincloud' else [0, 100]
                ax[1, gi].set_ylim(ylims)
                ax[1, gi].set_ylabel('Choice Rate (%)')
                ax[1, gi].spines['top'].set_visible(False)
                ax[1, gi].spines['right'].set_visible(False)
                if subplot_title and gi == 0:
                    ax[1, gi].annotate(subplot_title[1],
                                    xy=(-.25, 1.15),
                                    xytext=(0, 0),
                                    xycoords='axes fraction',
                                    textcoords='offset points',
                                    ha='left',
                                    va='top',
                                    fontweight='bold')
                plt.subplots_adjust(left=0.2)

            #Metaplot settings
            fig.tight_layout()
            if not os.path.exists('RL/plots/model_behaviours'):
                os.makedirs('RL/plots/model_behaviours')
            save_name = f"{m}_model_behaviours_supplemental.png" if plot_type == 'raincloud' else f"{m}_model_behaviours.png"
            fig.savefig(os.path.join('RL','plots', 'model_behaviours', save_name))
            fig.savefig(os.path.join('RL','plots', 'model_behaviours', save_name.replace('.png','.svg')), format='svg')

            #Close figure
            plt.close()

    def plot_fits_by_run_number(self, fit_data: dict) -> None:
        
        """
        Plots the negative log likelihood of model fits by run number for each model.

        Parameters
        ----------
        fit_data : dict
            A dictionary where keys are model names and values are DataFrames containing fit data with columns 'run' and 'fit'.

        Returns
        -------
        None
        """

        min_run, max_run = fit_data[list(fit_data.keys())[0]]['run'].min()+1, fit_data[list(fit_data.keys())[0]]['run'].max()+1
        best_run = {model: [] for model in fit_data}
        best_fits = {model: {f'{run}': [] for run in range(min_run, max_run+1)} for model in fit_data}
        for model in fit_data:
            model_data = fit_data[model].copy()
            model_data['run'] += 1
            for run in range(min_run, max_run+1):
                run_data = model_data[model_data['run'] <= run].reset_index(drop=True)
                run_sums = run_data.groupby('run').agg({'fit': 'mean'}).reset_index()
                best_fits[model][f'{run}'] = run_sums['fit'].min()
            best_run[model] = list(best_fits[model].keys())[list(best_fits[model].values()).index(min(best_fits[model].values()))]
        average_best_run = int(np.median([int(best_run[model]) for model in best_run]))

        #Create a subplot for each model and plot the fits by run number
        num_subplots = len(fit_data)
        number_of_columns = min(num_subplots, 5)
        number_of_rows = int(np.ceil(num_subplots / number_of_columns))

        fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(5*number_of_columns, 5*number_of_rows))
        for n, model in enumerate(best_fits):
            row, col = n//5, n%5
            if len(best_fits) == 1:
                ax = axs
            else:
                ax = axs[row, col] if len(best_fits) > 5 else axs[n]
            ax.plot([int(x) for x in best_fits[model].keys()], list(best_fits[model].values()), marker='o')
            ax.axvline(x=average_best_run, color='red', linestyle='--', alpha=.5)
            ax.set_title(model)
            ax.set_xlabel('Run Number')
            ax.set_ylabel('Negative Log Likelihood')

        plt.tight_layout()
        plt.savefig('RL/plots/fit-by-runs.png')
        plt.savefig('RL/plots/fit-by-runs.svg', format='svg')

        #Close figure
        plt.close()

    def rename_models(self, model_name: str) -> str:

        """
        Renames the model name for better readability in plots.

        Parameters
        ----------
        model_name : str
            The original model name to be renamed.
        
        Returns
        -------
        str
            The renamed model name.
        """

        return model_name.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning').replace('2012','')

    def plot_model_fits(self, confusion_matrix: pd.DataFrame) -> None:

        """
        Plots the model fits as a confusion matrix.

        Parameters
        ----------
        confusion_matrix : pd.DataFrame
            A DataFrame representing the confusion matrix where rows are true models and columns are fitted models.

        Returns
        -------
        None
        """

        confusion_matrix = confusion_matrix - confusion_matrix.min().min()
        confusion_matrix = confusion_matrix / confusion_matrix.max().max() * 200 - 100

        cmap = plt.get_cmap('RdBu')
        green = [0.596078431372549, 0.8117647058823529, 0.5843137254901961, 1]
        red = [0.9411764705882353, 0.5490196078431373, 0.5529411764705883, 1]
        white = [1, 1, 1, 1]
        custom_cmap = cmap.from_list('custom_cmap', [green, white, red], 20)
        
        #Plot confusion matrix
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        cax = ax.matshow(confusion_matrix, cmap=custom_cmap, alpha=1)
        cbar = ax.figure.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(confusion_matrix.columns)))
        ax.set_yticks(np.arange(len(confusion_matrix.index)))
        ax.set_xticklabels([self.rename_models(col) for col in confusion_matrix.columns])
        ax.set_yticklabels([self.rename_models(ind) for ind in confusion_matrix.index])
        cbar.set_label('Normalized BIC')
        for i in range(len(confusion_matrix.index)):
            for j in range(len(confusion_matrix.columns)):
                if confusion_matrix.iloc[i, j] == confusion_matrix.iloc[:, j].min():
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=2))
        for j in range(len(confusion_matrix.columns)):
            bfi = confusion_matrix.iloc[:, j].idxmin()
            i = confusion_matrix.index.get_loc(bfi)
            ax.text(j, i, np.round(confusion_matrix.iloc[i, j],2), ha='center', va='center', color='black', fontweight='bold')
        for i in range(len(confusion_matrix.index)):
            for j in range(len(confusion_matrix.columns)):
                if confusion_matrix.iloc[i, j] != confusion_matrix.iloc[:, j].min():
                    ax.text(j, i, np.round(confusion_matrix.iloc[i, j],2), ha='center', va='center', color='black')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='left')
        plt.tight_layout()
        plt.savefig(f'RL/plots/model_recovery.png')
        plt.savefig(f'RL/plots/model_recovery.svg', format='svg')

        #Close figure
        plt.close()

    def plot_parameter_fits(self,
                            models: list,
                            fit_data: dict,
                            fixed: dict = None,
                            bounds: dict = None,
                            alpha: float = 0.75) -> None:

        """
        Plots the parameter fits for each model, comparing true parameters with fitted parameters.

        Parameters
        ----------
        models : list
            List of model names to plot.
        fit_data : dict
            A dictionary where keys are model names and values are DataFrames containing fit data with columns 'participant' and model parameters.
        fixed : dict, optional
            A dictionary specifying which parameters are fixed for each model. Default is None.
        bounds : dict, optional
            A dictionary specifying the bounds for each parameter in each model. Default is None.
        alpha : float, optional
            Transparency level for the scatter plots. Default is 0.75.

        Returns
        -------
        None
        """

        #Create a dictionary with model being keys and pd.dataframe empty as value
        fit_results = {model: [] for model in models}
        for model in models:
            model_data = fit_data[model]
            for run_params in model_data['participant']:
                data_name = run_params.replace('[','').replace(']','')
                true_parameters = pd.read_csv(f'RL/data/generated/{data_name}/{data_name}_generated_parameters.csv')
                fit_parameters = pd.DataFrame(model_data[model_data['participant']==run_params].values[0][4:]).T
                fit_parameters.columns = true_parameters.columns

                true_parameters['Model'] = model
                fit_parameters['Model'] = model
                true_parameters['fit_type'] = 'True'
                fit_parameters['fit_type'] = 'Fit'
                combined_parameters = pd.concat([true_parameters, fit_parameters])
                combined_parameters = combined_parameters[['Model', 'fit_type'] + [col for col in combined_parameters.columns if col not in ['Model', 'fit_type']]]

                if isinstance(fit_results[model], pd.DataFrame):
                    fit_results[model] = pd.concat([fit_results[model], combined_parameters])
                else:
                    fit_results[model] = combined_parameters            

        number_columns = np.max([len(fit_results[model].columns)-2 for model in models])
        fig, axs = plt.subplots(len(models), number_columns, figsize=(5*number_columns, 5*len(models)), squeeze=False)
        for mi, model in enumerate(models):
            model_bounds = RLModel(model, fixed=fixed, bounds=bounds).get_bounds()
            for i, parameter in enumerate(fit_results[model].columns[2:]):
                parameter_name = parameter.replace('_', ' ').replace('lr', 'learning rate').title()
                true = fit_results[model][fit_results[model]['fit_type']=='True'][parameter]
                fit = fit_results[model][fit_results[model]['fit_type']=='Fit'][parameter]
                axs[mi, i].scatter(true, fit, s=5, alpha=alpha)
                axs[mi, i].plot(model_bounds[parameter], model_bounds[parameter], '--', color='grey', alpha=alpha)
                axs[mi, i].set_title(f"{parameter_name.title().replace('Learning Rate','LR')}", loc='left', fontsize=12)
                if mi == len(models)-1:
                    axs[mi, i].set_xlabel('True')
                else: 
                    axs[mi, i].set_xlabel('')
                if i == 0:
                    axs[mi, i].set_ylabel(f"{model.split('+')[0].replace('2012','').replace('ActorCritic', 'Actor Critic')}\nFit")
                else:
                    axs[mi, i].set_ylabel('')
                axs[mi, i].set_xlim(model_bounds[parameter])
                axs[mi, i].set_ylim(model_bounds[parameter])
                axs[mi, i].set_xticks([model_bounds[parameter][0], model_bounds[parameter][1]])
                axs[mi, i].set_yticks([model_bounds[parameter][0], model_bounds[parameter][1]])
                #set tick label fontsize to 14
                axs[mi, i].tick_params(axis='both', which='major', labelsize=14)
                axs[mi, i].tick_params(axis='both', which='minor', labelsize=14)
                axs[mi, i].spines['top'].set_visible(False)
                axs[mi, i].spines['right'].set_visible(False)                
                if i == len(fit_results[model].columns[2:])-1 and i != (number_columns-1):
                    for j in range(i+1, number_columns):
                        axs[mi, j].axis('off')    
                    
        plt.tight_layout()
        if not os.path.exists('RL/plots/correlations'):
            os.makedirs('RL/plots/correlations')
        plt.savefig(f'RL/plots/correlations/recovery_correlation_plot.png')
        plt.savefig(f'RL/plots/correlations/recovery_correlation_plot.svg', format='svg')

        #Close figure
        plt.close()

    def plot_parameter_data(self, save_name: str, model_data: pd.DataFrame = None, plot_type: str = 'raincloud') -> None:

        """
        Create raincloud plots of the data

        Parameters
        ----------
        save_name : str
            The name to save the plot as

        Returns
        -------
        None
        """

        #Set data specific parameters
        if 'model-fits' in save_name:
            if model_data is None:
                data = self.model_fits
            else:
                data = model_data
                data.set_index('pain_group', inplace=True)

        condition_name = 'pain_group'
        condition_values = ['no pain', 'acute pain', 'chronic pain']
        x_values = np.arange(1, len(condition_values)+1).tolist()
        x_labels = ['No\nPain', 'Acute\nPain', 'Chronic\nPain']
        plot_labels = data.columns[3:]
        num_subplots = len(plot_labels)
        
        number_of_columns = min(num_subplots, 3)
        number_of_rows = int(np.ceil(num_subplots / number_of_columns))
        fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=(5*number_of_columns, 5*number_of_rows))
        for pi, parameter in enumerate(plot_labels):
            if parameter != '':
                group_data = data[parameter].reset_index()
            else:
                group_data = data.reset_index()
            group_data[condition_name] = pd.Categorical(group_data[condition_name], condition_values)
            group_data = group_data.sort_values(condition_name)

            #Compute t-statistic
            _, t_scores = self.compute_n_and_t(group_data, condition_name)

            #Get descriptive statistics for the group
            group_data = group_data.set_index(condition_name)[parameter].astype(float)
            if parameter not in ['novel_value', 'mixing_factor', 'valence_factor', 'weighting_factor']: # Exclude parameters that are not to be log-transformed
                if group_data.min() <= 0: 
                    group_data = group_data - group_data.min() + 1  # Shift the parameter to be positive if it has non-positive values
                group_data = np.log(group_data)  # Log-transform the parameter to reduce skewness

            #Create plot
            row, col = pi//3, pi%3
            if num_subplots == 1:
                ax = axs
            else:
                ax = axs[row, col] if num_subplots > 3 else axs[pi]
            if plot_type == 'raincloud':
                self.raincloud_plot(data=group_data, ax=ax, t_scores=t_scores)
            elif plot_type == 'bar':
                self.group_bar_plot(data=group_data, ax=ax, t_scores=t_scores)
            else:
                raise ValueError(f"Invalid plot type {plot_type}. Choose either 'raincloud' or 'bar'.")

            #Create horizontal line for the mean the same width
            ax.set_xticks(x_values, x_labels)
            ax.set_xlabel('')
            y_label = parameter.replace('_', ' ').replace('lr', 'learning rate').title()
            y_label = f'Log {y_label}' if parameter not in ['novel_value', 'mixing_factor', 'valence_factor', 'weighting_factor'] else f'{y_label}'
            ax.set_ylabel(y_label)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
        
        for pi in range(num_subplots, (number_of_rows*number_of_columns)):
            row, col = pi//3, pi%3
            ax = axs[row, col] if num_subplots > 3 else axs[pi]
            fig.delaxes(ax)

        #Save the plot
        if not os.path.exists('RL/plots/fits'):
            os.makedirs('RL/plots/fits')
        save_name = f'{save_name}_supplemental' if plot_type == 'raincloud' else save_name
        plt.savefig(f'RL/plots/fits/{save_name}.png')
        plt.savefig(f'RL/plots/fits/{save_name}.svg', format='svg')

        #Close figure
        plt.close()

    def raincloud_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.75) -> None:
            
            """
            Create a raincloud plot of the data

            Parameters
            ----------
            data : DataFrame
                The data to be plotted
            ax : Axes
                The axes to plot the data on
            t_scores : list
                The t-scores for each group
            alpha : float
                The transparency of the scatter plot

            Returns
            -------
            None
            """
            
            #Set parameters
            if data.index.nunique() == 2:
                colors = self.get_colors('condition_2')
            elif data.index.nunique() == 3:
                colors = self.get_colors('group')
            else:
                colors = self.get_colors('condition')

            #Set index name
            data.index.name = 'code'
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data.columns = ['score']

            #Create a violin plot of the data for each level
            wide_data = data.reset_index().pivot(columns='code', values='score')
            wide_data = wide_data[data.index.unique()]
            wide_list = [wide_data[code].dropna() for code in wide_data.columns]
            vp = ax.violinplot(wide_list, showmeans=False, showmedians=False, showextrema=False)
            
            for bi, b in enumerate(vp['bodies']):
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], m, np.inf)
                b.set_color(colors[bi])

            #Add jittered scatter plot of the choice rate for each column
            for factor_index, factor in enumerate(data.index.unique()):
                x = np.random.normal([factor_index+1-.2]*data.loc[factor].shape[0], 0.02)
                ax.scatter(x+.02, data.loc[factor], color=colors[factor_index], s=10, alpha=alpha)
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_data = data.groupby('code').mean()
            CIs = data.groupby('code').sem()['score'] * t_scores

            #Draw rectangle for each symbol that rerpesents the top and bottom of the 95% CI that has no fill and a black outline
            for factor_index, factor in enumerate(data.index.unique()):
                ax.add_patch(plt.Rectangle((factor_index+1-0.4, (mean_data.loc[factor] - CIs.loc[factor])['score']), 0.8, 2*CIs.loc[factor], fill=None, edgecolor='darkgrey'))
                ax.hlines(mean_data.loc[factor], factor_index+1-0.4, factor_index+1+0.4, color='darkgrey')            

    def group_bar_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.75) -> None:

            """
            Create a raincloud plot of the data

            Parameters
            ----------
            data : DataFrame
                The data to be plotted
            ax : Axes
                The axes to plot the data on
            t_scores : list
                The t-scores for each group
            alpha : float
                The transparency of the scatter plot
            """

            #Set colors
            if data.index.nunique() == 2:
                colors = self.get_colors('condition_2')
            elif data.index.nunique() == 3:
                colors = self.get_colors('group')
            else:
                colors = self.get_colors('condition')

            #Set index name
            data.index.name = 'code'
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data.columns = ['score']
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_data = data.groupby('code').mean()
            mean_data = mean_data.reindex(['no pain', 'acute pain', 'chronic pain'])
            mean_data = mean_data.dropna()
            CIs = data.groupby('code').sem()['score'] 
            CIs = CIs.reindex(['no pain', 'acute pain', 'chronic pain'])
            CIs = CIs.dropna()
            CIs = CIs * t_scores

            #Add barplot with CIs
            ax.bar(np.arange(1,len(mean_data['score'])+1), mean_data['score'], yerr=CIs, color=colors, alpha=alpha, capsize=5, ecolor='dimgrey')                        

    def bar_plot(self, data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.75) -> None:
            
            """
            Create a raincloud plot of the data

            Parameters
            ----------
            data : DataFrame
                The data to be plotted
            ax : Axes
                The axes to plot the data on
            t_scores : list
                The t-scores for each group
            alpha : float
                The transparency of the scatter plot
            """

            #Set colors
            if data.index.nunique() == 2:
                colors = self.get_colors('condition_2')
            elif data.index.nunique() == 3:
                colors = self.get_colors('group')
            else:
                colors = self.get_colors('condition')

            #Set index name
            data.index.name = 'code'
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data.columns = ['score']
            
            #Compute the mean and 95% CIs for the choice rate for each symbol
            mean_data = data.groupby('code').mean()
            mean_data = mean_data.reindex(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'])
            mean_data = mean_data.dropna()
            CIs = data.groupby('code').sem()['score'] 
            CIs = CIs.reindex(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'])
            CIs = CIs.dropna()
            CIs = CIs * t_scores

            #Add barplot with CIs
            ax.bar(np.arange(1,len(mean_data['score'])+1), mean_data['score'], yerr=CIs, color=colors, alpha=alpha, capsize=5, ecolor='dimgrey')                        

    def compute_n_and_t(self, data: pd.DataFrame, splitting_column: str) -> tuple:

        """
        Compute the sample size and t-score for each group

        Parameters
        ----------
        data : DataFrame
            The data to be analyzed
        splitting_column : str
            The column to split the data by

        Returns
        -------
        sample_sizes : list
            The sample sizes for each group
        t_scores : list
            The t-scores for each group
        """

        #Reset index to allow access to all columns
        data = data.reset_index()

        #Compute the sample size and t-score for each group
        if splitting_column == None:
            sample_sizes = data.shape[0]
            t_scores = stats.t.ppf(0.975, sample_sizes-1)
        else:
            sample_sizes = [data[data[splitting_column] == group].shape[0] for group in data[splitting_column].unique()]
            t_scores = [stats.t.ppf(0.975, s-1) for s in sample_sizes]

        return sample_sizes, t_scores

    def plot_fit_distributions(self, fit_data: dict) -> None:

        """
        Plots the distribution of BICs for each model fit.

        Parameters
        ----------
        fit_data : dict
            A dictionary where keys are model names and values are DataFrames containing fit data with columns 'fit'.
        
        Returns
        -------
        None
        """

        models = fit_data.keys()
        fig, ax = plt.subplots(1, len(models), figsize=(5*len(models), 5))
        for i, model in enumerate(models):
            model_data = fit_data[model]['fit'].values
            
            #Fit transformation
            number_samples = 480
            number_params = len(fit_data[model].columns) - 4
            BICs = np.log(number_samples) * number_params + 2 * model_data

            #Determine distribution
            mu, std = np.mean(BICs), np.std(BICs)
            x = np.linspace(min(BICs), max(BICs), 100)
            y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / std)**2) # TODO: Redo distrubution

            #Compute kurtosis and skewness
            kurtosis = pd.Series(BICs).kurtosis()
            skewness = pd.Series(BICs).skew()

            #Plot histogram and fitted normal distribution
            if len(models) > 1:
                ax[i].hist(BICs, bins=10, density=True, alpha=0.33, color='green')
                ax[i].axvline(np.mean(BICs), color='green', linestyle='dashed', linewidth=1)
                ax[i].plot(x, y, color='green', linewidth=2, label=None)
                ax[i].set_title(f'{model}\nKurtosis: {kurtosis:.2f}, Skewness: {skewness:.2f}')
                ax[i].set_xlabel('BIC')
                ax[i].set_ylabel('Proportion')
            else:
                ax.hist(BICs, bins=10, density=True, alpha=0.33, color='green')
                ax.axvline(np.mean(BICs), color='green', linestyle='dashed', linewidth=1)
                ax.plot(x, y, color='green', linewidth=2, label=None)
                ax.set_title(f'{model}\nKurtosis: {kurtosis:.2f}, Skewness: {skewness:.2f}')
                ax.set_xlabel('BIC')
                ax.set_ylabel('Proportion')
        plt.tight_layout()
        plt.savefig('RL/plots/model_fits_distributions.png')
        plt.savefig('RL/plots/model_fits_distributions.svg', format='svg')

        #Close figure
        plt.close()

    def plot_model_comparisons(self, fits: pd.DataFrame, diffs = pd.DataFrame, save_name: str = 'model_comparisons') -> None:

        """
        Plots the model fits as a bar plot with colors representing the fit values.

        Parameters
        ----------
        fits : pd.DataFrame
            A DataFrame where the index is the model names and the values are the fit values (e.g., BIC).
        diffs : pd.DataFrame, optional
            A DataFrame where the index is the model names and the values are the differences from the best fitting model. It contains mean and 95% CIs.
        save_name : str, optional
            The name to save the plot as. Default is 'model_comparisons'.

        Returns
        -------
        None
        """

        fits = fits.loc['full'][:-1]
        min_val, max_val = fits.min().min(), fits.max().max()
        range_val = max_val - min_val

        fit_diffs = fits - fits.min()
        fit_diffs = fit_diffs[fit_diffs != 0].dropna()

        model_dict = {'QLearning': 'Q-Learning', 'ActorCritic': 'Actor Critic', 'Relative': 'Relative', 'Advantage': 'Advantage','Hybrid2012': 'Hybrid'}
        models = [model_dict.get(model.split('+')[0], model) for model in fits.index]

        #Create a bar plot of the model fits
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        #Raw values
        ax[0].bar(np.arange(1, len(fits.index)+1), fits.values, color=self.get_colors('group')[0], alpha=0.9)
        for i, v in enumerate(fits.values):
            ax[0].text(i+1, v + (range_val*0.02), str(int(np.round(v, 0))), color='darkgrey', ha='center', fontweight='bold', fontsize=14)
        lower_bar_index = np.where(fits.values == fits.values.min())[0][0]
        lowest_bar_value = fits.values.min()
        ax[0].add_patch(plt.Rectangle((lower_bar_index+0.6, min_val-(range_val*.5)), .8, lowest_bar_value-(min_val-(range_val*.5)), fill=False, edgecolor='black', lw=2, alpha=.6))

        ax[0].set_xticks(np.arange(1, len(fits.index)+1), models)
        plt.setp(ax[0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')    
        ax[0].set_ylabel('BIC')
        ax[0].set_xlabel('')
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_ylim([min_val-(range_val*.5), max_val+(range_val*0.1)])
        ax[0].annotate('A', xy=(-0.1, 1.05), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', ha='right', va='top', fontweight='bold', fontsize=16)

        #Normalized values
        diffs = diffs.T
        models = [model_dict.get(model.split('+')[0], model) for model in diffs.index]
        min_val, max_val = (diffs['mean']-diffs['ci']).min(), (diffs['mean']+diffs['ci']).max()
        range_val = max_val - min_val
        ax[1].bar(np.arange(1, len(diffs.index)+1), diffs['mean'].values, color=self.get_colors('group')[0], alpha=0.9)
        ax[1].errorbar(np.arange(1, len(diffs.index)+1), diffs['mean'].values, yerr=diffs['ci'].values, fmt='none', ecolor='dimgray', capsize=5, elinewidth=2, alpha=1.0)
        lower_bar_index = np.where(diffs.values == diffs.values.min())[0][0]
        lowest_bar_value = diffs.values.min()

        ax[1].set_xticks(np.arange(1, len(diffs.index)+1), models)
        plt.setp(ax[1].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')    
        ax[1].set_ylabel('BIC Difference')
        ax[1].set_xlabel('')
        ax[1].spines['top'].set_visible(False)
        ax[1].spines['right'].set_visible(False)
        #ax[1].set_ylim([0, max_val+(range_val*0.1)])
        ax[1].axhline(0, color='dimgray', linestyle='--', linewidth=1, alpha=0.5)
        ax[1].annotate('B', xy=(-0.1, 1.05), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', ha='right', va='top', fontweight='bold', fontsize=16)

        plt.tight_layout()

        #Save the plot
        plt.savefig(f'RL/plots/{save_name}.png')
        
        #Close figure
        plt.close()

    def get_colors(self, color_type: str = None) -> dict:

        """
        Returns a dictionary of colors for different plot types.

        Parameters
        ----------
        color_type : str, optional
            The type of colors to return. Options are 'group', 'condition', or 'condition_2'. If None, returns all colors.

        Returns
        -------
        dict, list
            A dictionary of colors for different plot types, or a list of colors for the specified type.
        """

        colors = {'group': ['#85A947', '#3E7B27', '#123524'],
                'condition': ['#095086', '#9BD2F2', '#ECA6A6', '#B00000', '#D3D3D3'],
                'condition_2': ['#095086', '#B00000']}
        
        if color_type is not None:
            return colors[color_type]
        else:
            return colors
        
    def plot_model_parameter_correlations(self, fit_data: dict, params_of_interest: dict) -> None:

        """
        Plots scatter plots comparing model parameters from different models.

        Parameters:
        - fit_data: dict of DataFrames, keyed by model name, each with a 'participant' column and parameter columns.
        - params_of_interest: dict mapping model names to the parameter of interest in each model.
        """

        # Remove any keys from params_of_interest that are not in fit_data
        params_of_interest = {model: param for model, param in params_of_interest.items() if model in fit_data and param in fit_data[model].columns}

        # Extract relevant columns for each model
        model_values = {
            model: fit_data[model][['participant', param]]
            for model, param in params_of_interest.items()
        }

        # Merge dataframes on participant
        model_keys = list(params_of_interest.keys())
        merged_df = model_values[model_keys[0]]
        for model in model_keys[1:]:
            merged_df = merged_df.merge(
                model_values[model], on='participant', suffixes=('', f'_{model}')
            )

        #Log-Transform the contextual learning rate if it exists
        if 'contextual_lr' in merged_df.columns:
            merged_df['contextual_lr'] = np.log(merged_df['contextual_lr'])

        # Create all pairwise combinations of parameters manually
        param_keys = list(params_of_interest.values())
        param_pairs = []
        for i in range(len(param_keys)):
            for j in range(i + 1, len(param_keys)):
                param_pairs.append((param_keys[i], param_keys[j]))

        # Create subplots
        n_pairs = len(param_pairs)
        corr_values = {}
        if n_pairs != 0:
            fig, axs = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 5))
            color = self.get_colors('condition')[0]

            if n_pairs == 1:
                axs = [axs]  # Make it iterable

            for i, (x_param, y_param) in enumerate(param_pairs):
                x = merged_df[x_param]
                y = merged_df[y_param]
                model_x = [k for k, v in params_of_interest.items() if v == x_param][0].split('+')[0]
                model_y = [k for k, v in params_of_interest.items() if v == y_param][0].split('+')[0]
                param_x = x_param.replace('_', ' ').replace('lr', 'learning rate').title()
                param_y = y_param.replace('_', ' ').replace('lr', 'learning rate').title()

                #Run the correlation
                corr, p_value = stats.pearsonr(x, y) 
                corr_values[f'{model_x}_{model_y}'] = f'{corr:.2f} ({p_value:.3f})'              

                #Plot the scatter plot
                axs[i].scatter(x, y, color=color, alpha=.5)
                axs[i].set_xlabel(f"{model_x.replace('2012','')}\n{param_x}")
                axs[i].set_ylabel(f"{model_y.replace('2012','')}\n{param_y}")
                if corr > 0:
                    axs[i].plot([x.min(), x.max()], [y.min(), y.max()], 'k--', lw=2, alpha=0.5)
                else:
                    axs[i].plot([x.min(), x.max()], [y.max(), y.min()], 'k--', lw=2, alpha=0.5)
                axs[i].set_xlim(x.min(), x.max())
                axs[i].set_ylim(y.min(), y.max())
                axs[i].set_xticks([x.min(), x.max()])
                axs[i].set_yticks([y.min(), y.max()])
                axs[i].set_xticklabels([f'{x.min():.2f}', f'{x.max():.2f}'])
                axs[i].set_yticklabels([f'{y.min():.2f}', f'{y.max():.2f}'])
                axs[i].spines['top'].set_visible(False)
                axs[i].spines['right'].set_visible(False)

            plt.tight_layout()
            plt.savefig('RL/plots/parameter_of_interest_comparisons.png', dpi=300)
            plt.close()

            print(corr_values)