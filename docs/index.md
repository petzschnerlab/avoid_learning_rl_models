---
hide:
- toc
---
# Avoidance Learning RL Models

Welcome to the avoidance learning reinforcement learning models repo. This repo was built for the PEAC lab to computationally model empirical data from the various avoidance learning tasks. This repo has the ability to fit RL models to empirical data and to validate these models through parameters and model recovery methods.  

<div style="text-align: center; margin-top: 1em;">
  <a href="Tutorials/RL_tutorial/" class="md-button md-button--primary">
    Jump to the tutorial!
  </a>
</div>

<b>Supported models</b><br>
&nbsp;&nbsp;&nbsp;&nbsp;QLearning, ActorCritic<br>
&nbsp;&nbsp;&nbsp;&nbsp;Relative, Advantage<br>
&nbsp;&nbsp;&nbsp;&nbsp;Hybrid2012, Hybrid2021<br>
&nbsp;&nbsp;&nbsp;&nbsp;StandardHybrid2012, StandardHybrid2021<br><br>

<b>Standard models </b><br>
Standard models as described in each reference, which introduces the model, with the addition of counterfactual learning rates.<br>
Hybrid models have alternatives versions without counterfactual learning rates: StandardHybrid2012, StandardHybrid2021
&nbsp;&nbsp;&nbsp;&nbsp;<b>QLearning</b>: Standard Q-Learning Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>ActorCritic</b>: Standard Actor-Critic Model<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Relative</b>: Standard Relative Model (Palminteri et al., 2015)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Advantage</b>: Simplified Relative Model (Williams et al., 2025)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Hybrid2012+bias</b>: Hybrid 2012 Model (Gold et al., 2012)<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>Hybrid2021+bias+decay</b>: Hybrid 2021 Model (Geana et al., 2021)<br><br>

<b>Optional Parameters</b><br>
You can add optional parameters to models by adding them to the model name using a + sign<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>+bias</b>: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid models<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>+novel</b>: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models<br>
&nbsp;&nbsp;&nbsp;&nbsp;<b>+decay</b>: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models

## Project Pipeline
This repo is one part of a project pipeline, which requires the coordination of multiple repos. Projects begin with a <b>task repo</b>, which is used to collect behavioural data from participants either locally or on Prolific. The collected data must then be pushed through a <b>data extraction repo</b> to prepare CSV files for analysis. These CSV files are used in <b>the analysis repo</b>, which creates a PDF report (`AL/reports`), ending the project pipeline. 

Optionally, you can run computational reinforcement learning models using the <b>modelling repo (this repo)</b>, and the results can be added to the report. This is a bit clunky because it requires a bit of back-and-forth between the analysis repo and this modelling repo. Specifically, the analysis repo must be run (with `load_models=False`) in order to create two CSV files that this repo needs (`AL/data/pain_learning_processed.csv` and `AL/data/pain_transfer_processed.csv`). These files can then be manually moved into this repo's data directory (`RL/data`). This repo can then be used to model the data, which will result in a newly constructed directory called `modelling` (`RL/modelling`). This folder can then be manually moved to the analysis repo as `AL/modelling`. Then you can re-run the analysis repo (with `load_models=True`) and the modelling results will be included in the PDF report. 

## Project Repos

### Task Repos
There exists several versions of the avoidance learning task. This package was built around two of these repos:

- [Version 1a](https://github.com/petzschnerlab/v1a_avoid_pain) 
- [Version 1b](https://github.com/petzschnerlab/v1b_avoid_paindepression)

There also exists other task repos that are likely compatible with this analysis code, but have never been tested:

- [Version 2](https://github.com/petzschnerlab/v2_avoid_paindepression_presample)
- [Version EEG](https://github.com/petzschnerlab/soma_avoid_eeg)

### Data Extraction Repo
There also exists some code that extracts the data collected by the task repos (which come as `.json` files) and formats it into a .csv file.

- [Data Extraction](https://github.com/petzschnerlab/avoid_learning_data_extraction)

### Analysis Repo
Next there is the analysis repo, which conducts statistics, creates plots, and build a PDF report of the main findings.

- [Analysis](https://github.com/petzschnerlab/avoid_learning_analysis)

### Computational Modelling Repo
Finally, there is a companion repo to the analysis repo, which fits computational reinforcement learning models to the data. 

- [RL Modelling](https://github.com/petzschnerlab/avoid_learning_rl_models)