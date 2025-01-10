import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

def plot_simulations(accuracy, prediction_errors, values, choice_rates, models, group):
    
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
        ax[0, i].set_title(m)
        ax[0, i].set_ylim([25, 110])
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
        ax[3, i].set_ylim([0, 100])
        if i == 0:
            ax[3, i].set_ylabel('Choice rate (%)')
        ax[3, i].spines['top'].set_visible(False)
        ax[3, i].spines['right'].set_visible(False)

    #Metaplot settings
    fig.tight_layout()
    #Add suptitle in top left 
    fig.suptitle(f"{group.title()}", x=0.001, y=.999, ha='left', fontsize=16)
    fig.savefig(os.path.join('SOMA_RL','plots',f"{group.replace(' ','')}_model_simulations.png"))
