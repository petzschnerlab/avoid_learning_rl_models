---
hide:
- toc
---

# PIPELINE OBJECT PARAMETERS

<table>
  <tr>
    <th><strong>NAME</strong></th>
    <th><strong>TYPE</strong></th>
    <th><strong>DEFAULT</strong></th>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>help</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Prints the help information for the package, including an overview and parameter descriptions.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>seed</strong></td>
    <td><strong>int | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Seed for random number generation. If None, no seed is set.</td>
  </tr>
</table>

# FIT MODE PARAMETERS

<table>
  <tr>
    <th><strong>NAME</strong></th>
    <th><strong>TYPE</strong></th>
    <th><strong>DEFAULT</strong></th>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>help</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Prints the help information for the package, including an overview and parameter descriptions.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>mode</strong></td>
    <td><strong>str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Mode of operation, either 'fit' or 'validation'. In FIT mode, models are fitted to empirical data. In VALIDATION mode, parameter recovery or model recovery is performed, depending on the recovery parameter. This is a required parameter, so there is no default value.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>models</strong></td>
    <td><strong>list[str] | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      List of models to fit.<br><br>
      Supported models:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;QLearning, ActorCritic<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Relative, Advantage<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2012, Hybrid2021, StandardHybrid2012, StandardHybrid2021<br><br>
      Standard models:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;QLearning: Standard Q-Learning Model<br>
      &nbsp;&nbsp;&nbsp;&nbsp;ActorCritic: Standard Actor-Critic Model<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Relative: Standard Relative Model (Palminteri et al., 2015)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Advantage: Simplified Relative Model (Williams et al., in prep)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)<br><br>
      Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+bias: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid2012, and Hybrid2021<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>learning_filename</strong></td>
    <td><strong>str | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Filename (and path) of the learning task data.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>transfer_filename</strong></td>
    <td><strong>str | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Filename (and path) of the transfer task data.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>random_params</strong></td>
    <td><strong>bool | str</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Mode of determining parameter starting points. Can be 'normal', 'random' or False. If 'normal' starting parameter values will be drawn from a normal distribution with the means being defined in the fixed parameter. The parameters will be cutoff at the bounds defined in the bounds parameter. If 'random' is selected, the parameters will be drawn from a uniform distribution between the bounds. If no fixed or bound parameters are provided, the default values will be used (found in the RLModel class).
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>fixed</strong></td>
    <td><strong>dict | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Optional fixed parameter values. For the FIT mode, these values will be used as the model parameters if random_params = False. If random_params = 'normal', these values will be used as the mean of the normal distribution. If random_params = 'random', these values will be ignored. For the VALIDATION mode, these values will be ignored if the parameter or fit_filename parameters are used. Otherwise, these values will be used in the same way as in the FIT mode.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>bounds</strong></td>
    <td><strong>dict | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Bounds that each parameter is cutoff at. Operates in both fitting and simulating data. It is a nested dictionairy in the form of {model: {parameter: tuple}}. The tuple are two floats representing the bottom and top bounds, e.g., bounds['QLearning']['factual_lr'] = (0.1, 0.99)
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>number_of_runs</strong></td>
    <td><strong>int</strong></td>
    <td><strong>1</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      How many times to fit each model per participant. Two outputs when fitting data is the fit_data.pkl and full_fit_data.pkl files. This works well with randomized starting points (e.g., random_params = 'random' or random_params = 'normal') because each run has a different set of starting parameters, which helps finding the best fit parameters. The fit_data.pkl file contains the best run for each model and participant, while the full_fit_data.pkl file contains all runs.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>generated</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Use generated data instead of empirical.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>multiprocessing</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Use multiprocessing for parallel computation.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>training</strong></td>
    <td><strong>str</strong></td>
    <td><strong>scipy</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Training backend to use [scipy, torch]. The pytorch backend is on beta testing. It works, but performs worse than the scipy backend. There has not yet been an investigation into why this is the case. If using the torch backend, the training_epochs and optimizer_lr parameters are used. These are ignored if the scipy backend is used.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>training_epochs</strong></td>
    <td><strong>int</strong></td>
    <td><strong>1000</strong></td>
  </tr>
  <tr>
    <td colspan="3">If using torch backend (training = 'torch'), this determines the number of training epochs.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>optimizer_lr</strong></td>
    <td><strong>float</strong></td>
    <td><strong>0.01</strong></td>
  </tr>
  <tr>
    <td colspan="3">If using torch backend (training = 'torch'), this is the learning rate for the ADAM optimizer (which is the only one implemented at this time).</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>number_of_participants</strong></td>
    <td><strong>int</strong></td>
    <td><strong>0</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      TEST PARAM. This parameter is used to cut your provided dataset down. It will cut it down to this number of participants. It will take the first N participants from the dataset, where N is the number of participants you inputted. If 0 is inputted (default), it will keep all participants. This is designed mostly for testing.
    </td>
  </tr>
</table>

# VALIDATE MODE PARAMETERS

<table>
  <tr>
    <th><strong>NAME</strong></th>
    <th><strong>TYPE</strong></th>
    <th><strong>DEFAULT</strong></th>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>help</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Prints the help information for the package, including an overview and parameter descriptions.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>mode</strong></td>
    <td><strong>str</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Mode of operation, either 'fit' or 'validation'. In FIT mode, models are fitted to empirical data. In VALIDATION mode, parameter recovery or model recovery is performed, depending on the recovery parameter. This is a required parameter, so there is no default value.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>recovery</strong></td>
    <td><strong>str</strong></td>
    <td><strong>parameter</strong></td>
  </tr>
  <tr>
    <td colspan="3">This parameter sets the recovery mode, it can be 'parameter' or 'model'. Parameter recovery is the process of generating data with known parameters for a given model, and then fitting that data with the same model to determine whether the parameters are recoverable. This can still use a list of models, but each model will only recover its own generated data. Model recovery is the process of generating data with known parameters for all given models. Each model is then fitted to all generated data (regardless of which model generated it) to test which model best fits data from every model. Ideally, the model that generated the data should be the best fit.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>models</strong></td>
    <td><strong>list[str] | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      List of models to fit.<br><br>
      Supported models:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;QLearning, ActorCritic<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Relative, Advantage<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2012, Hybrid2021, StandardHybrid2012, StandardHybrid2021<br><br>
      Standard models:<br>
      &nbsp;&nbsp;&nbsp;&nbsp;QLearning: Standard Q-Learning Model<br>
      &nbsp;&nbsp;&nbsp;&nbsp;ActorCritic: Standard Actor-Critic Model<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Relative: Standard Relative Model (Palminteri et al., 2015)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Advantage: Simplified Relative Model (Williams et al., in prep)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2012+bias: Standard Hybrid 2012 Model (Gold et al., 2012)<br>
      &nbsp;&nbsp;&nbsp;&nbsp;Hybrid2021+bias+decay: Standard Hybrid 2021 Model (Geana et al., 2021)<br><br>
      Optional Parameters: You can add optional parameters to models by adding them to the model name using a + sign<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+bias: Adds a valence bias to the model (e.g. Hybrid2012+bias), only usable with Hybrid2012, and Hybrid2021<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+novel: Adds a free parameter for the novel stimulus (e.g. QLearning+novel), useable with all models<br>
      &nbsp;&nbsp;&nbsp;&nbsp;+decay: Adds a decay parameter to the model (e.g. QLearning+decay), useable with all models
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>learning_filename</strong></td>
    <td><strong>str | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Filename (and path) of the learning task data.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>transfer_filename</strong></td>
    <td><strong>str | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">Filename (and path) of the transfer task data.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>random_params</strong></td>
    <td><strong>bool | str</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Mode of determining parameter starting points. Can be 'normal', 'random' or False. If 'normal' starting parameter values will be drawn from a normal distribution with the means being defined in the fixed parameter. The parameters will be cutoff at the bounds defined in the bounds parameter. If 'random' is selected, the parameters will be drawn from a uniform distribution between the bounds. If no fixed or bound parameters are provided, the default values will be used (found in the RLModel class).
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>fixed</strong></td>
    <td><strong>dict | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Optional fixed parameter values. For the FIT mode, these values will be used as the model parameters if random_params = False. If random_params = 'normal', these values will be used as the mean of the normal distribution. If random_params = 'random', these values will be ignored. For the VALIDATION mode, these values will be ignored if the parameter or fit_filename parameters are used. Otherwise, these values will be used in the same way as in the FIT mode.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>bounds</strong></td>
    <td><strong>dict | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Bounds that each parameter is cutoff at. Operates in both fitting and simulating data. It is a nested dictionairy in the form of {model: {parameter: tuple}}. The tuple are two floats representing the bottom and top bounds, e.g., bounds['QLearning']['factual_lr'] = (0.1, 0.99)
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>parameters</strong></td>
    <td><strong>dict | list[dict]</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      A nested dictionary where the first level is the model name, which then is a dictionary of the model parameters with their values. These values will be used to generate data using the model, and the random_params variable will be ignored. This parameter conflicts with the fit_filename parameter, so if both are provided, you will receive an error. The intended use of this parameter is to run a specific model with set parameters. Note that this is not the same as the fixed parameter, which is used as priors (the mean) when using random_params='normal'. However, if this parameter is not provided, and random_params = False, then the fixed parameter will be used as the model parameters.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>fit_filename</strong></td>
    <td><strong>str | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      A nested dictionary where the first level is the model name, which then is a dictionary of the model parameters with their values. This is used to load pre-fitted model parameters from a file, which would have been saved as fit_data.pkl when running the fit mode. This parameter overrides the parameters parameter, so all information provided there is also relevant for this parameter. This file will use participant IDs to determine the model parameters for each participant, so it is important that the same data is being used in recovery as was used in the fit mode.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>task_design</strong></td>
    <td><strong>dict | None</strong></td>
    <td><strong>None</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      This parameter defines the task parameters (e.g., trials) to be run. This is an alternative to the learning_filename and transfer_filename parameters, which instead use the predifined task designs for each participant. This parameter is a nested dictionary with the following structure: The highest level must have 'learning_phase' and 'transfer_phase' as keys. The learning_phase key should then include a dict with 'number_of_trials' and 'number_of_blocks' as keys. The transfer_phase key should then include a dict with 'number_of_trials' or 'times_repeated' as keys. These keys in the transfer_phase are mutually exclusive, meaning you can only use one of them at a time. If both are provided, number_of_trials will be used. In the transfer_phase, the 'number_of_trials' key is used to define the number of trials in the transfer phase, and the 'times_repeated' key is instead used to define how many times all pairs of stimuli are repeated (there exists 36 pairs).
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>datasets_to_generate</strong></td>
    <td><strong>int</strong></td>
    <td><strong>1</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      The number of participants to generate if using the task_design parameter. This will be overriden with the number of participants you have if you use the learning_filename and transfer_filename parameters.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>number_of_runs</strong></td>
    <td><strong>int</strong></td>
    <td><strong>1</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      How many times to fit each model per participant. Two outputs when fitting data is the fit_data.pkl and full_fit_data.pkl files. This works well with randomized starting points (e.g., random_params = 'random' or random_params = 'normal') because each run has a different set of starting parameters, which helps finding the best fit parameters. The fit_data.pkl file contains the best run for each model and participant, while the full_fit_data.pkl file contains all runs.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>number_of_participants</strong></td>
    <td><strong>int</strong></td>
    <td><strong>0</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      TEST PARAM. This parameter is used to cut your provided dataset down. It will cut it down to this number of participants. It will take the first N participants from the dataset, where N is the number of participants you inputted. If 0 is inputted (default), it will keep all participants. This is designed mostly for testing.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>generate_data</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>True</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      This boolean will determine whether you will generate new data using the models. The intended use of this parameter is to toggle off data generation if you have already generated data. For example, if you generated data when validating using recovery='parameter' (i.e., generate_data=True), and now want to run recovery='model' using the same generated data (i.e., generate_data=False).
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>clear_data</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>True</strong></td>
  </tr>
  <tr>
    <td colspan="3">Whether to clear previous simulated data before generating new. This will be ignored if generate_data=False so not to delete the data you are planning to use.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>multiprocessing</strong></td>
    <td><strong>bool</strong></td>
    <td><strong>False</strong></td>
  </tr>
  <tr>
    <td colspan="3">Use multiprocessing for parallel computation.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>training</strong></td>
    <td><strong>str</strong></td>
    <td><strong>scipy</strong></td>
  </tr>
  <tr>
    <td colspan="3">
      Training backend to use [scipy, torch]. The pytorch backend is on beta testing. It works, but performs worse than the scipy backend. There has not yet been an investigation into why this is the case. If using the torch backend, the training_epochs and optimizer_lr parameters are used. These are ignored if the scipy backend is used.
    </td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>training_epochs</strong></td>
    <td><strong>int</strong></td>
    <td><strong>1000</strong></td>
  </tr>
  <tr>
    <td colspan="3">If using torch backend (training = 'torch'), this determines the number of training epochs.</td>
  </tr>
  <tr style="background-color:#f0f0f0;">
    <td><strong>optimizer_lr</strong></td>
    <td><strong>float</strong></td>
    <td><strong>0.01</strong></td>
  </tr>
  <tr>
    <td colspan="3">If using torch backend (training = 'torch'), this is the learning rate for the ADAM optimizer (which is the only one implemented at this time).</td>
  </tr>
</table>
