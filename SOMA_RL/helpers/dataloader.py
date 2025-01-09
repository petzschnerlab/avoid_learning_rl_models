import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, learning_filename, transfer_filename, number_of_participants=0):

        '''
        A class to load and preprocess the data from the learning and transfer phases of the experiment.

        Parameters
        ----------
        learning_filename : str
            The filename of the learning phase data.
        transfer_filename : str
            The filename of the transfer phase data.
        number_of_participants : int
            The number of participants to include in the analysis. If 0, all participants are included.
            Mostly for debugging purposes
        '''

        learning_data = pd.read_csv(learning_filename)
        transfer_data = pd.read_csv(transfer_filename)

        #Reorganize data so that left stim is always better
        learning_data['stim_order'] = learning_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
        learning_data['reward_L'] = learning_data.apply(lambda x: x['feedback_L']/10 if x['stim_order'] else x['feedback_R']/10, axis=1)
        learning_data['reward_R'] = learning_data.apply(lambda x: x['feedback_R']/10 if x['stim_order'] else x['feedback_L']/10, axis=1)
        learning_data['action'] = learning_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)

        learning_data = learning_data[['participant_id', 'group_code', 'symbol_names', 'reward_L', 'reward_R', 'action']]
        learning_data['symbol_names'] = learning_data['symbol_names'].replace({'Reward1': 'State AB', 'Reward2': 'State CD', 'Punish1': 'State EF', 'Punish2': 'State GH'})
        learning_data.columns = ['participant', 'pain_group', 'state', 'reward_L', 'reward_R', 'action']

        transfer_data['stim_order'] = transfer_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
        transfer_data['symbol_L_name'] = transfer_data['symbol_L_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '75P1': 'E', '25P1': 'F', '75P2': 'G', '25P2': 'H', 'Zero': 'N'})
        transfer_data['symbol_R_name'] = transfer_data['symbol_R_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '75P1': 'E', '25P1': 'F', '75P2': 'G', '25P2': 'H', 'Zero': 'N'})
        transfer_data['state'] = transfer_data.apply(lambda x: f"State {x['symbol_L_name']}{x['symbol_R_name']}" if x['stim_order'] else f"State {x['symbol_R_name']}{x['symbol_L_name']}", axis=1)
        transfer_data['action'] = transfer_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)
        
        transfer_data = transfer_data[['participant_id', 'group_code', 'state', 'action']]
        transfer_data.columns = ['participant', 'pain_group', 'state', 'action']

        #DEBUGGING CLEANUP
        #Cut the dataframe to n participants: #TODO:This is for debugging, remove it later
        if number_of_participants > 0:
            p_indices = learning_data['participant'].unique()[:number_of_participants]
            learning_data = learning_data[learning_data['participant'].isin(p_indices)]
            transfer_data = transfer_data[transfer_data['participant'].isin(p_indices)]

        self.learning_data = learning_data
        self.transfer_data = transfer_data

    def filter_participant_data(self, participant):
        participant_learning_data = self.learning_data[self.learning_data['participant'] == participant].copy()
        participant_transfer_data = self.transfer_data[self.transfer_data['participant'] == participant].copy()
        return {'learning': participant_learning_data, 'transfer': participant_transfer_data}

    def get_participant_ids(self):
        return self.learning_data['participant'].unique()

    def get_data(self):
        return self.learning_data, self.transfer_data