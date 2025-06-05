import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self, learning_filename: str, transfer_filename: str, number_of_participants: int = 0, reduced: bool = True, generated: bool = False):

        """
        Initializes the DataLoader with learning and transfer data.

        Parameters:
        -----------
        learning_filename : str
            Path to the learning data CSV file.
        transfer_filename : str
            Path to the transfer data CSV file.
        number_of_participants : int, optional
            Number of participants to limit the data to (default is 0, which means no limit).
        reduced : bool, optional
            If True, reduces the data to essential columns (default is True).
        generated : bool, optional
            If True, indicates that the data is generated (default is False).

        Returns:
        --------
        None
        
        """

        learning_data = pd.read_csv(learning_filename)
        transfer_data = pd.read_csv(transfer_filename)

        #Reorganize data so that left stim is always better
        if generated:
            learning_data['group_code'] = 'simulated'
            learning_data = learning_data.rename(columns={'state_id': 'symbol_names'})
            learning_data['rewards'] = learning_data['rewards'].apply(lambda x: np.array(eval(x)))
            learning_data['reward_L'] = learning_data['rewards'].apply(lambda x: x[0])
            learning_data['reward_R'] = learning_data['rewards'].apply(lambda x: x[1])

            transfer_data['group_code'] = 'simulated'
            transfer_data = transfer_data.rename(columns={'state_id': 'state'})
        else:
            learning_data['stim_order'] = learning_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
            learning_data['reward_L'] = learning_data.apply(lambda x: x['feedback_L']/10 if x['stim_order'] else x['feedback_R']/10, axis=1)
            learning_data['reward_R'] = learning_data.apply(lambda x: x['feedback_R']/10 if x['stim_order'] else x['feedback_L']/10, axis=1)
            learning_data['action'] = learning_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)
            learning_data['symbol_names'] = learning_data['symbol_names'].replace({'Reward1': 'State AB', 'Reward2': 'State CD', 'Punish1': 'State EF', 'Punish2': 'State GH'})

            transfer_data['stim_order'] = transfer_data.apply(lambda x: x['symbol_L_value'] > x['symbol_R_value'], axis=1)
            transfer_data['symbol_R_name'] = transfer_data['symbol_R_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '25P1': 'E', '75P1': 'F', '25P2': 'G', '75P2': 'H', 'Zero': 'N'})
            transfer_data['symbol_L_name'] = transfer_data['symbol_L_name'].replace({'75R1': 'A', '25R1': 'B', '75R2': 'C', '25R2': 'D', '25P1': 'E', '75P1': 'F', '25P2': 'G', '75P2': 'H', 'Zero': 'N'})
            transfer_data['state'] = transfer_data.apply(lambda x: f"State {x['symbol_L_name']}{x['symbol_R_name']}" if x['stim_order'] else f"State {x['symbol_R_name']}{x['symbol_L_name']}", axis=1)
            transfer_data['action'] = transfer_data.apply(lambda x: x['choice_made'] if x['stim_order'] else np.abs(x['choice_made']-1), axis=1)

        if reduced:
            learning_data = learning_data[['participant_id', 'group_code', 'symbol_names', 'reward_L', 'reward_R', 'action']]
            transfer_data = transfer_data[['participant_id', 'group_code', 'state', 'action']]
        learning_data.columns = [colname.replace('participant_id', 'participant').replace('group_code', 'pain_group').replace('symbol_names', 'state') for colname in learning_data.columns]
        transfer_data.columns = [colname.replace('participant_id', 'participant').replace('group_code', 'pain_group') for colname in transfer_data.columns]

        #This is for testing:
        if number_of_participants > 0:
            p_indices = learning_data['participant'].unique()[:number_of_participants]
            learning_data = learning_data[learning_data['participant'].isin(p_indices)]
            transfer_data = transfer_data[transfer_data['participant'].isin(p_indices)]

        self.learning_data = learning_data
        self.transfer_data = transfer_data

    def filter_participant_data(self, participant: str) -> None:
        """
        Filters the learning and transfer data for a specific participant.

        Parameters:
        -----------
        participant : str
            The participant ID to filter the data for.

        Returns:
        --------
        None
        """

        self.learning_data = self.learning_data[self.learning_data['participant'] == participant]
        self.transfer_data = self.transfer_data[self.transfer_data['participant'] == participant]

    def get_participant_ids(self) -> np.ndarray:
        """
        Gets the unique participant IDs from the learning data.

        Returns:
        --------
        np.ndarray
            An array of unique participant IDs.
        """

        return self.learning_data['participant'].unique()
    
    def get_group_ids(self) -> np.ndarray:
        """
        Gets the unique pain group IDs from the learning data.

        Returns:
        --------
        np.ndarray
            An array of unique pain group IDs.
        
        """
        return self.learning_data['pain_group'].unique()
    
    def get_num_samples(self) -> int:

        """
        Gets the total number of samples across both learning and transfer data.

        Returns:
        --------
        int
            The total number of samples.

        """

        return self.learning_data.shape[0] + self.transfer_data.shape[0]
    
    def get_num_samples_by_group(self, group_id: str) -> int:

        """
        Gets the total number of samples for a specific pain group across both learning and transfer data.

        Parameters:
        -----------
        group_id : str
            The pain group ID to filter the data for.

        Returns:
        --------
        int
            The total number of samples for the specified pain group.
        """

        num_learning_samples = self.learning_data[self.learning_data['pain_group'] == group_id].shape[0]
        num_transfer_samples = self.transfer_data[self.transfer_data['pain_group'] == group_id].shape[0]
        return num_learning_samples+num_transfer_samples
    
    def get_num_trials(self) -> tuple:
        
        """
        Gets the number of trials in the learning and transfer data.

        Returns:
        --------

        tuple
            A tuple containing the number of trials in the learning data and the transfer data.
        """

        return self.learning_data.shape[0], self.transfer_data.shape[0]

    def get_data(self) -> tuple:
        """
        Returns the learning and transfer data as pandas DataFrames.

        Returns:
        --------
        tuple
            A tuple containing the learning data and transfer data DataFrames.
        """

        return self.learning_data, self.transfer_data
    
    def get_data_dict(self) -> dict:

        """
        Returns the learning and transfer data as a dictionary.

        Returns:
        --------
        dict
            A dictionary containing the learning and transfer data DataFrames.
        """

        return {'learning': self.learning_data, 'transfer': self.transfer_data}