import os
import copy
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from models.rl_models import RLModel

def plot_simulations(accuracy, prediction_errors, values, choice_rates, models, group, dataloader=None):
    
    if dataloader is not None:

        #Get empirical learning curves 
        empirical_data = copy.copy(dataloader.get_data()[0])
        empirical_data = empirical_data[empirical_data['pain_group'] == group]
        emp_accuracy = empirical_data.groupby(['context_val_name','trial_number'], observed=False)['accuracy'].mean().reset_index()

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

    colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']
    bi_colors = ['#B2DF8A', '#FB9A99']
    val_colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C']
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
            ax[0, i].fill_between(context_accuracy.index, context_accuracy - context_CIs, context_accuracy + context_CIs, alpha=0.2, color=bi_colors[ci], edgecolor='none')
            ax[0, i].plot(context_accuracy, color=bi_colors[ci], alpha = .8, label=context.replace('Loss Avoid', 'Punish'))
            if dataloader is not None:
                ax[0, i].plot(emp_accuracy[emp_accuracy['context_val_name'] == context]['trial_number'], emp_accuracy[emp_accuracy['context_val_name'] == context]['accuracy'], color=bi_colors[ci], linestyle='dashed', alpha=.5)

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
            ax[1, i].fill_between(context_pe.index, context_pe - context_CIs, context_pe + context_CIs, alpha=0.2, color=bi_colors[ci], edgecolor='none')
            ax[1, i].plot(context_pe, color=bi_colors[ci], alpha = .8, label=context.replace('Loss Avoid', 'Punish'))
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
                ax[2, i].fill_between(context_values.index, context_values - context_CIs, context_values + context_CIs, alpha=0.2, color=val_colors[ci*2+vi], edgecolor='none')
                ax[2, i].plot(context_values, color=val_colors[ci*2+vi], alpha = .8, label=['High Reward', 'Low Reward', 'Low Punish', 'High Punish'][ci*2+vi])
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
        ax[3, i].bar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), color=colors, alpha = .5)
        ax[3, i].errorbar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem()* stats.t.ppf(0.975, number_of_participants-1), fmt='.', color='grey')
        if dataloader is not None:
            ax[3, i].scatter(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], list(emp_choice_rates.values()), color='grey', marker='D', alpha=.5)
        ax[3, i].set_ylim([0, 100])
        if i == 0:
            ax[3, i].set_ylabel('Choice Rate (%)')
        ax[3, i].spines['top'].set_visible(False)
        ax[3, i].spines['right'].set_visible(False)

    #Metaplot settings
    fig.tight_layout()
    fig.suptitle(f"{group.title()}", x=0.001, y=.999, ha='left', fontsize=16)
    fig.savefig(os.path.join('SOMA_RL','plots',f"{group.replace(' ','')}_model_simulations.png"))
    fig.savefig(os.path.join('SOMA_RL','plots',f"{group.replace(' ','')}_model_simulations.svg"), format='svg')

def plot_simulations_behaviours(accuracy, choice_rates, models, groups, dataloader=None):
    colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']
    bi_colors = ['#B2DF8A', '#FB9A99']

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
            model_accuracy = accuracy[m][group].groupby(['context', 'trial_total', 'run'], observed=False).mean().reset_index()
            model_accuracy['context'] = pd.Categorical(model_accuracy['context'], categories=['Reward', 'Loss Avoid'], ordered=True)

            for ci, context in enumerate(['Reward', 'Loss Avoid']):
                CIs = model_accuracy.groupby(['context', 'trial_total'], observed=False)['accuracy'].sem() * stats.t.ppf(0.975, number_of_participants-1)
                averaged_accuracy = model_accuracy.groupby(['context', 'trial_total'], observed=False).mean().reset_index()
                context_accuracy = averaged_accuracy[averaged_accuracy['context'] == context]['accuracy'].reset_index(drop=True).astype(float) * 100
                context_CIs = CIs[CIs.index.get_level_values('context') == context].reset_index(drop=True) * 100
                
                ax[0, gi].fill_between(context_accuracy.index, context_accuracy - context_CIs, context_accuracy + context_CIs, alpha=0.2, color=bi_colors[ci], edgecolor='none')
                ax[0, gi].plot(context_accuracy, color=bi_colors[ci], alpha=0.8, label=context.replace('Loss Avoid', 'Punish'))

                if dataloader is not None:
                    trials = emp_accuracy_group[group][emp_accuracy_group[group]['context_val_name'] == context]['trial_number']
                    accuracies = emp_accuracy_group[group][emp_accuracy_group[group]['context_val_name'] == context]['accuracy']
                    ax[0, gi].plot(trials, accuracies, color=bi_colors[ci], linestyle='dashed', alpha=0.5)
            
            ax[0, gi].set_title(f'{group.title()}')
            ax[0, gi].set_ylim([25, 100])
            if gi == len(groups) - 1:
                ax[0, gi].legend(loc='lower right', frameon=False)
            ax[0, gi].set_ylabel('Accuracy (%)')
            ax[0, gi].set_xlabel('Trial')
            ax[0, gi].spines['top'].set_visible(False)
            ax[0, gi].spines['right'].set_visible(False)

            # Plot choice rates
            _, t_scores = compute_n_and_t(choice_rates[m][group], None)
            choice_data_long = choice_rates[m][group].stack().reset_index()[['level_1', 0]]
            choice_data_long.columns = ['stimulus', 'choice_rate']
            choice_data_long['stimulus'] = choice_data_long['stimulus'].replace({'A': 'High\nReward', 'B': 'Low\nReward', 'E': 'Low\nPunish', 'F': 'High\nPunish', 'N': 'Novel'})
            choice_data_long.set_index('stimulus', inplace=True)
            raincloud_plot(data=choice_data_long, ax=ax[1, gi], t_scores=t_scores)
            ax[1, gi].errorbar(np.arange(1,choice_data_long.index.nunique()+1), choice_rates[m][group].mean(axis=0), yerr=choice_rates[m][group].sem()*stats.t.ppf(0.975, number_of_participants-1), fmt='.', color='grey')
            
            if dataloader is not None:
                ax[1, gi].scatter(np.arange(1,choice_data_long.index.nunique()+1), list(emp_choice_groups[group].values()), color='grey', marker='D', alpha=0.5)

            #Set x-ticks and labels for the choice rate plot
            ax[1, gi].set_xticks(np.arange(1,choice_data_long.index.nunique()+1), ['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'])
            ax[1, gi].set_ylim([0, 100])
            ax[1, gi].set_ylabel('Choice Rate (%)')
            ax[1, gi].spines['top'].set_visible(False)
            ax[1, gi].spines['right'].set_visible(False)

        #Metaplot settings
        fig.tight_layout()
        fig.savefig(os.path.join('SOMA_RL','plots', 'model_behaviours', f"{m}_model_behaviours.png"))
        fig.savefig(os.path.join('SOMA_RL','plots', 'model_behaviours', f"{m}_model_behaviours.svg"), format='svg')

def plot_fits_by_run_number(fit_data):

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
    plt.savefig('SOMA_RL/plots/fit-by-runs.png')
    plt.savefig('SOMA_RL/plots/fit-by-runs.svg', format='svg')

def rename_models(model_name):
    return model_name.split('+')[0].replace('Hybrid2', 'Hybrid 2').replace('ActorCritic', 'Actor Critic').replace('QLearning', 'Q Learning')

def plot_model_fits(confusion_matrix):

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
    ax.set_xticklabels([rename_models(col) for col in confusion_matrix.columns])
    ax.set_yticklabels([rename_models(ind) for ind in confusion_matrix.index])
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
    plt.savefig(f'SOMA_RL/plots/model_recovery.png')
    plt.savefig(f'SOMA_RL/plots/model_recovery.svg', format='svg')

def plot_parameter_fits(models, fit_data, fixed=None, bounds=None):
    #Create a dictionary with model being keys and pd.dataframe empty as value
    fit_results = {model: [] for model in models}
    for model in models:
        model_data = fit_data[model]
        for run_params in model_data['participant']:
            data_name = run_params.replace('[','').replace(']','')
            true_parameters = pd.read_csv(f'SOMA_RL/data/generated/{data_name}/{data_name}_generated_parameters.csv')
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

    for model in models:
        model_bounds = RLModel(model, fixed=fixed, bounds=bounds).get_bounds()
        fig, axs = plt.subplots(1, len(fit_results[model].columns)-2, figsize=(5*len(fit_results[model].columns)-2, 5))
        for i, parameter in enumerate(fit_results[model].columns[2:]):
            true = fit_results[model][fit_results[model]['fit_type']=='True'][parameter]
            fit = fit_results[model][fit_results[model]['fit_type']=='Fit'][parameter]
            axs[i].scatter(true, fit)
            r = np.round(np.corrcoef(true.to_numpy().astype(float), fit.to_numpy().astype(float))[0,1], 2)
            axs[i].plot(model_bounds[parameter], model_bounds[parameter], '--', color='grey', alpha=0.5)
            axs[i].set_title(f"{parameter}, r={r}")
            axs[i].set_xlabel('True')
            axs[i].set_ylabel('Fit')
            axs[i].set_xlim(model_bounds[parameter])
            axs[i].set_ylim(model_bounds[parameter])

        fig.suptitle(f'{model} Correlation Plot')
        plt.tight_layout()
        if not os.path.exists('SOMA_RL/plots/correlations'):
            os.makedirs('SOMA_RL/plots/correlations')
        plt.savefig(f'SOMA_RL/plots/correlations/{model}_correlation_plot.png')
        plt.savefig(f'SOMA_RL/plots/correlations/{model}_correlation_plot.svg', format='svg')

def plot_parameter_rainclouds(save_name: str, model_data: pd.DataFrame = None) -> None:

    """
    Create raincloud plots of the data

    Parameters
    ----------
    save_name : str
        The name to save the plot as

    Returns (External)
    ------------------
    Image: PNG
        A plot of the raincloud plots
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
    x_labels = ['No Pain', 'Acute Pain', 'Chronic Pain']
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
        _, t_scores = compute_n_and_t(group_data, condition_name)

        #Get descriptive statistics for the group
        group_data = group_data.set_index(condition_name)[parameter].astype(float)
        if parameter not in ['novel_value', 'mixing_factor', 'valence_factor', 'weighing_factor']: # Exclude parameters that are not to be log-transformed
            if group_data.min() <= 0: 
                group_data = group_data - group_data.min() + 1  # Shift the parameter to be positive if it has non-positive values
            group_data = np.log(group_data)  # Log-transform the parameter to reduce skewness

        #Create plot
        row, col = pi//3, pi%3
        if num_subplots == 1:
            ax = axs
        else:
            ax = axs[row, col] if num_subplots > 3 else axs[pi]
        raincloud_plot(data=group_data, ax=ax, t_scores=t_scores)

        #Create horizontal line for the mean the same width
        ax.set_xticks(x_values, x_labels)
        ax.set_xlabel('')
        y_label = parameter.replace('_', ' ').replace('lr', 'learning rate').title()
        y_label = f'Log-Transformed {y_label}' if parameter not in ['novel_value', 'mixing_factor', 'valence_factor'] else f'{y_label}'
        ax.set_ylabel(y_label)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
    
    for pi in range(num_subplots, (number_of_rows*number_of_columns)):
        row, col = pi//3, pi%3
        ax = axs[row, col] if num_subplots > 3 else axs[pi]
        fig.delaxes(ax)

    #Save the plot
    plt.savefig(f'SOMA_RL/plots/fits/{save_name}.png')
    plt.savefig(f'SOMA_RL/plots/fits/{save_name}.svg', format='svg')

    #Close figure
    plt.close()

def raincloud_plot(data: pd.DataFrame, ax: plt.axes, t_scores: list[float], alpha: float=0.25) -> None:
        
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
        
        #Set parameters
        if data.index.nunique() == 2:
            colors = ['#B2DF8A', '#FB9A99']
        elif data.index.nunique() == 3:
            colors = ['#B2DF8A', '#FFD92F', '#FB9A99']
        else:
            colors = ['#33A02C', '#B2DF8A', '#FB9A99', '#E31A1C', '#D3D3D3']

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

def compute_n_and_t(data: pd.DataFrame, splitting_column: str) -> tuple:

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

def plot_fit_distributions(fit_data):
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
    plt.savefig('SOMA_RL/plots/model_fits_distributions.png')
    plt.savefig('SOMA_RL/plots/model_fits_distributions.svg', format='svg')
    plt.close()