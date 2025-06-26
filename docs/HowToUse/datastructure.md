---
hide:
- toc
---
# Dataset Description Avoidance Learning RL Models

This dataset contains trial-level data from an avoidance learning task. This data was first collected using a task repo and then processed using the data extraction repo. Finally, it the data was processed using the analysis repo, which we will use here (see project pipeline on [home](../index.md)). So, these data are the output of the analysis repo. There are two different datasets, one for the learning phase (e.g., `pain_learning_processed.csv`) and one for the transfer phase (e.g., `pain_learning_processed.csv`). The details of these files will be outlined below, however, now all columns are needed for this repo. The analysis script exports all columns at the end of processing, regardless of whether it will here be important or not.

## Learning Sample Data Table

| participant_id      | trial_type      | symbol_L_name | symbol_R_name | symbol_L_value | symbol_R_value | feedback_L | feedback_R | context_val | context_val_name | symbol_names | symbol_name | trial_number | binned_trial | choice_made | accuracy | rt   | duration     | group_code   | intensity | unpleasant | interference | composite_pain | age | sex  |
|---------------------|-----------------|----------------|----------------|----------------|----------------|------------|------------|--------------|-------------------|---------------|--------------|---------------|----------------|--------------|----------|------|---------------|--------------|-----------|-------------|----------------|----------------|------|------|
| 63f58712a2c1bd62fa7bb352 | learning-trials | 75P1           | 25P1           | 1              | 2              | 0          | 0          | -1           | Loss Avoid         | Punish1       | Punish       | 1             | Early          | 1            | 100      | 1758 | 1 – 5 years   | chronic pain | 6.45      | 6.1         | 6.45           | 6.33           | 29   | Male |
| 63f58712a2c1bd62fa7bb352 | learning-trials | 25P2           | 75P2           | 2              | 1              | -10        | -10        | -1           | Loss Avoid         | Punish2       | Punish       | 1             | Early          | 0            | 100      | 1759 | 1 – 5 years   | chronic pain | 6.45      | 6.1         | 6.45           | 6.33           | 29   | Male |
| 63f58712a2c1bd62fa7bb352 | learning-trials | 75R1           | 25R1           | 4              | 3              | 10         | 0          | 1            | Reward             | Reward1       | Reward       | 1             | Early          | 1            | 0        | 1479 | 1 – 5 years   | chronic pain | 6.45      | 6.1         | 6.45           | 6.33           | 29   | Male |

## Transfer Sample Data Table

| participant_id       | trial_index | rt   | symbol_L_name | symbol_R_name | symbol_L_value | symbol_R_value | feedback_L | feedback_R | choice_made | symbol_chosen | symbol_ignored | paired_symbols | context_val | context_val_name | accuracy | group_code   | composite_pain | duration     | intensity | unpleasant | interference | age | sex  |
|----------------------|-------------|------|----------------|----------------|----------------|----------------|------------|------------|--------------|----------------|----------------|----------------|-------------|------------------|----------|--------------|----------------|--------------|-----------|-------------|---------------|-----|------|
| 63f58712a2c1bd62fa7bb352 | 137         | 1753 | Zero           | 25R2           | 0              | 3              | 1          | 1          | 1            | 3              | 0              | 3_0           | 1           | Reward           | 100      | chronic pain | 6.333          | 1 – 5 years | 6.45      | 6.1         | 6.45          | 29  | Male |
| 63f58712a2c1bd62fa7bb352 | 138         | 1849 | 25R1           | 25P2           | 3              | 2              | -1         | 1          | 0            | 2              | 3              | 3_2           | -1          | Loss Avoid       | 0        | chronic pain | 6.333          | 1 – 5 years | 6.45      | 6.1         | 6.45          | 29  | Male |
| 63f58712a2c1bd62fa7bb352 | 142         | 1665 | Zero           | 25P1           | 0              | 2              | 0          | 1          | 1            | 2              | 0              | 2_0           | 0           | 0                | 100      | chronic pain | 6.333          | 1 – 5 years | 6.45      | 6.1         | 6.45          | 29  | Male |

## Column Descriptions

| Column Name            | Description |
|------------------------|-------------|
| `participant_id`       | Unique identifier for each participant. |
| `trial_type`           | Type of trial, either `"learning-trials"` or `"probe"` |
| `symbol_L_name`        | Identifier for the symbol presented on the left. May be `75R1, 75R2, 25R1, 25R2, 25P1, 25P2, 75P1, 75P2, Zero (neutral in transfer phase)`. |
| `symbol_R_name`        | Identifier for the symbol presented on the right. May be `75R1, 75R2, 25R1, 25R2, 25P1, 25P2, 75P1, 75P2, Zero (neutral in transfer phase)`. |
| `symbol_L_value`       | Numerical value assigned to the left symbol on that trial. `0=Novel`, `1=75P`, `2=25P`, `3=25R`, `4=75R`|
| `symbol_R_value`       | Numerical value assigned to the right symbol on that trial.  `0=Novel`, `1=75P`, `2=25P`, `3=25R`, `4=75R`|
| `feedback_L`           | Feedback value associated with the left symbol. May be `-10`, `0`, or `10`. This will be empty for the transfer phase. |
| `feedback_R`           | Feedback value associated with the right symbol. May be `-10`, `0`, or `10`. This will be empty for the transfer phase. |
| `context_val`          | Encoded task context value:  `1` (reward), `-1` (punishment)|
| `context_val_name`     | Human-readable description of `context_val`, e.g., `"Reward"` or `"Loss Avoid"`. |
| `symbol_names`         | Context names, dissociating between the different pairs within each context (`Reward1`, `Reward2`, `Punish1`, `Punish2`). |
| `symbol_name`          | Context names, collapsing across similar pairs (`Reward`, `Punish`). |
| `trial_number`         | Trial number within a given symbol's presentation history (e.g., trial number for `75R1`). |
| `binned_trial`         | Trial bin based on its position in the task (`Trials 1-6=Early`, `Trials 7-12=Mid-Early`, `Trials 13-18=Mid-Late`, `Trials 19-24=Late`). |
| `choice_made`          | Binary indicator for the participant's choice: `1` for right, `0` for left. |
| `symbol_chosen`        | The numerical value of the symbol that was chosen on the trial. |
| `symbol_ignored`       | The numerical value of the symbol that was not chosen on the trial. |
| `paired_symbols`       | Identifiers for each unique pair, respective of their symbol values. e.g., `4_1` is `75R` versus `75P`|
| `accuracy`             | Accuracy score for the trial, `100` for correct and `0` for incorrect. |
| `rt`                   | Reaction time in milliseconds. |
| `duration`             | Self-report string describing current pain state, e.g., `"I am not in pain"` or `"1 – 5 years"`. |
| `group_code`           | Participant group code (typically categorical or condition label): `"no pain"`, `"acute pain"`, or `"chronic pain"`. |
| `intensity`            | Self-reported pain intensity rating. |
| `unpleasant`           | Self-reported unpleasantness rating. |
| `interference`         | Self-reported rating of pain interference with daily life. |
| `composite_pain`       | Composite pain score (e.g., average of intensity, unpleasantness, and interference ratings). |
| `age`                  | Participant's age in years. |
| `sex`                  | Participant’s reported sex (e.g., `"Female"`, `"Male"`, `"Other"`). |

