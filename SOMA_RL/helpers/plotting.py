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
        ax[3, i].errorbar(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], choice_rates[m].mean(axis=0), yerr=choice_rates[m].sem(), fmt='.', color='grey')
        if dataloader is not None:
            ax[3, i].scatter(['High\nReward', 'Low\nReward', 'Low\nPunish', 'High\nPunish', 'Novel'], list(emp_choice_rates.values()), color='grey', marker='D', alpha=.5)
        ax[3, i].set_ylim([0, 100])
        if i == 0:
            ax[3, i].set_ylabel('Choice rate (%)')
        ax[3, i].spines['top'].set_visible(False)
        ax[3, i].spines['right'].set_visible(False)

    #Metaplot settings
    fig.tight_layout()
    fig.suptitle(f"{group.title()}", x=0.001, y=.999, ha='left', fontsize=16)
    fig.savefig(os.path.join('SOMA_RL','plots',f"{group.replace(' ','')}_model_simulations.png"))


def plot_fits_by_run_number(fit_data_path):
    #Load pickle file fit_data_path
    with open(fit_data_path, 'rb') as f:
        fit_data = pickle.load(f)

    min_run, max_run = fit_data[list(fit_data.keys())[0]]['run'].min()+1, fit_data[list(fit_data.keys())[0]]['run'].max()+1
    best_run = {model: [] for model in fit_data}
    best_fits = {model: {f'Run {run}': [] for run in range(min_run, max_run)} for model in fit_data}
    for model in fit_data:
        model_data = fit_data[model].copy()
        model_data['run'] += 1
        for run in range(min_run, max_run):
            #Find data where run equals or is less than run
            run_data = model_data[model_data['run'] <= run].reset_index(drop=True)
            run_sums = run_data.groupby('run').agg('sum').reset_index()
            best_fits[model][f'Run {run}'] = run_sums['fit'].min()
        best_run[model] = list(best_fits[model].keys())[list(best_fits[model].values()).index(min(best_fits[model].values()))]
    average_best_run = f"Run {int(np.ceil(np.mean([int(best_run[model].split(' ')[1]) for model in best_run])))}"

    #Create a subplot for each model and plot the fits by run number
    fig, axs = plt.subplots(len(best_fits)//4, len(best_fits)//3, figsize=(5*(len(best_fits)//3), (5*(len(best_fits)//4))))
    for n, model in enumerate(best_fits):
        row, col = n//4, n%4
        ax = axs[row, col] if len(best_fits) > 1 else axs
        ax.plot(list(best_fits[model].keys()), list(best_fits[model].values()), marker='o')
        ax.axvline(x=average_best_run, color='red', linestyle='--', alpha=.5)
        ax.set_title(model)
        ax.set_xlabel('Run Number')
        ax.set_ylabel('Best Fit')
    fig.text(0.01, 0.001, f'Red dashed line indicates the averaged run where the models reached their best fits.', ha='left')

    plt.tight_layout()
    #Save plot 
    plt.savefig(fit_data_path.replace('.pkl', '.png'))

def plot_generative_fits(models, fit_data, fixed=None, bounds=None):
    #Create a dictionary with model being keys and pd.dataframe empty as value
    fit_results = {model: [] for model in models}
    for model in models:
        model_data = fit_data[model]
        for run_params in model_data['participant']:
            
            true_parameters = pd.read_csv(f'SOMA_RL/data/generated/{model}_{run_params}/{model}_{run_params}_generated_parameters.csv')
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
        #Plot correlation plots, new figure for each model, subplot for each parameter

        bounds = RLModel(model, fixed=fixed, bounds=bounds).get_bounds()
        fig, axs = plt.subplots(1, len(fit_results[model].columns)-2, figsize=(5*len(fit_results[model].columns)-2, 5))
        for i, parameter in enumerate(fit_results[model].columns[2:]):
            true = fit_results[model][fit_results[model]['fit_type']=='True'][parameter]
            fit = fit_results[model][fit_results[model]['fit_type']=='Fit'][parameter]
            axs[i].scatter(true, fit)
            r = np.round(np.corrcoef(true.to_numpy().astype(float), fit.to_numpy().astype(float))[0,1], 2)
            axs[i].plot(bounds[parameter], bounds[parameter], '--', color='grey', alpha=0.5)
            axs[i].set_title(f"{parameter}, r={r}")
            axs[i].set_xlabel('True')
            axs[i].set_ylabel('Fit')
            axs[i].set_xlim(bounds[parameter])
            axs[i].set_ylim(bounds[parameter])

        fig.suptitle(f'{model} Correlation Plot')
        plt.tight_layout()
        plt.savefig(f'SOMA_RL/plots/{model}_correlation_plot.png')