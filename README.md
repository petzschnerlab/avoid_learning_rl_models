- **[COMPLETE] Model Fitting**: I know you are currently using fixed starting points. I think to make sure that our parameter estimates are really reliable we need to see hwo stable the are across different starting points: “A key limitation of optimizers like fmincon is that they are only guaranteed to find local minima which are not guaranteed to be the global minima corresponding to the best fitting parameters. One way to mitigate this issue is to run the fitting procedure multiple times with random initial conditions, recording the best fitting log-likelihood for each run. The best fitting parameters are then the parameters corresponding to the run with the highest log-likelihood. There is no hard-and-fast rule for knowing how many starting points to use in a given situation, besides the fact that more complex models will require more starting points. Thus, it must be determined empirically in each case. One way to validate the number of starting points is by plotting the best likelihood score as a function of the number of starting points. As the number of initial conditions increases, the best fitting likelihood (and corresponding parameters) will improve up to an asymptote close to the true maximum of the function.

<p align="center">
    <img src=".\SOMA_RL\model%20results\Standard%20Models%20+%20Novel%20for%2010%20Runs\full_fit_data.png" width="600" height="400" />
</p>

- **Parameter Recovery**: Can we recover the parameters for each model. First, simulate fake data with known parameter values. Next, fit the model to this fake data to try to ‘recover’ the parameters. Finally, compare the recovered parameters to their true values. In a perfect world the simulated and recovered parameters will be tightly correlated, with no bias. If there is only a weak correlation between simulated and recovered parameters and/or a significant bias, then this is an indication that there is either a bug in your code (which from our own experience we suggest is fairly likely) or the experiment is underpowered to assess this model. Read the tips on page 14/ 15 of the Collins paper
    - **STATUS**:
        - Completed data generation (maybe expand to run random parameters so that each model generated multiple parameters)
        - TODO: Expand script to then fit data with corresponding model (random params n times, where n is determined by results of Model Fitting)
        - TODO: Create correlation plots of different parameters and true parameters 

<p align="center">
    <img src=".\SOMA_RL\plots\correlations\QLearning_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\QLearning+novel_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\ActorCritic_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\ActorCritic+novel_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Relative_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Relative+novel_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\wRelative+bias+decay_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\wRelative+bias+decay+novel_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Hybrid2012+bias_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Hybrid2012+bias+novel_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Hybrid2021+bias+decay_correlation_plot.png" width="800" height="200" />
    <img src=".\SOMA_RL\plots\correlations\Hybrid2021+bias+decay+novel_correlation_plot.png" width="800" height="200" />
</p>

- **Model Comparison**: I would add a Bayesian model comparison to our data analysis using the VB Toolbox by Lionel Rigoux.
    - **TODO**: Implement this with PyMC

- **Model Recovery**: Can we recover the model from which we simulated the data. Here we are simulating data from all models (with a range of parameter values as in the case of parameter recovery) and then fitting that data with all models to determine the extent to which fake data generated from model A is best fit by model A as opposed to model B. This process can be summarized in a confusion matrix (see Figure 5 in Collins paper) that quantifies the probability that each model is the best fit to data generated from the other models. In a perfect world the confusion matrix will be the identity matrix, but in practice this is not always the case e.g. (Wilson & Niv, 2012).
    - **TODO**: Create analysis script to compare model fitting of simulated data

- **Winning Model Validation**: Does the winning model predict the actual behavior: A better method to validate a model is to simulate it with the fit parameter values (Palminteri et al., 2017; Nassar & Frank, 2016; Navarro, 2018). You should then analyze the simulated data in the same way you analyzed the empirical data, to verify that all important behavioral effects are qualitatively and quantitatively captured by the simulations with the fit parameters. For example, if you observe a qualitative difference between two conditions empirically, the model should reproduce it. Likewise if a learning curve reaches a quantitative asymptote of 0.7, simulations shouldn’t reach a vastly different one. Some researchers analyze the posterior prediction of the model conditioned on the past history, instead of simulated data. In our previous notation they evaluate the likelihood of choice ct given past data, d1:t−1, where the past data includes choices made by the subject, not choices made by the model, p(ct|d1:t−1, st, θm, m). In some cases, this approach leads to very similar results to simulations, because simulations sample choices based on a very similar probability, where the past data, d1:t−1, include choices made by the model. However, it can also be dramatically different if the path of actions sampled by the participant is widely different from the paths likely to be selected by the model (leading to very different past histories).
    - **TODO**: Create statistical analysis script (or plug it back into SOMA_AL?) to analyze and plot the effects

- **Final Figures**: 
    - **Figure 1**: Task Structure
    - **Figure 2**: Learning across groups (behavior and winning model prediction)
    - **Figure 3**: Test across groups (behavior and winning model prediction)
    - **Figure 4**: Model comparison
    - **Supplementary Figures**: Parameter Recovery
    - **Supplementary Figures**: Confusion Matrix
    - **Supplementary Figure**: Single subject fit (example of trial trajectory)


## Other Considerations

### Parameter values from the literature

- Hybrid 2021: Geana et al 2021 (Healthy Controls)
  - Q LR: .49
  - Actor LR: .33
  - Critic LR: .48
  - Beta: 15.72
  - Reward Discount (d): .33
  - Mixing Param: .69
  - Decay: .08
  - Noise: .04

- Hybrid 2012: Gold et al 2012 (Healthy Controls)
  - Only mixing parameter was reported
  - Mixing Parameter (c): 0.7

- Relative: Palminteri et al 2015
  - Beta: 21.52 +/- 5.95 (SEM)
  - Factual LR: .19 +/- .02 (SEM)
  - Counterfactual LR: .15 +/- .02 (SEM)
  - Context LR: 0.33 +/- .07 (SEM)

- QLearning: Palminteri et al., 2015
  - Beta: 17.4 +/- 5.92 (SEM)
  - Factual LR: 0.28 +/- .02 (SEM)
  - Counterfactual LR: 0.18 +/- .02 (SEM)

Actor Critic
  - Using parameters from Geana et al., 2021