import os
import time
import numpy as np
import tqdm

def mp_run_fit(args):
    rl_pipeline = args[0]
    rl_pipeline.run_rl_fit(args[1:])

def mp_run_simulations(args):
    rl_pipeline = args[0]
    if 'generate_data' in str(args[-1]):
        rl_pipeline.run_simulations(args[1:-1], generate_data=args[-1].split('=')[-1])
    else:
        rl_pipeline.run_simulations(args[1:])

def mp_progress(num_files, filepath='SOMA_RL/fits/temp', divide_by=1, multiply_by=1, progress_bar=True):
    n_files = 0
    last_count = 0
    start_file_count = len(os.listdir(filepath))
    if progress_bar:
        loop = tqdm.tqdm(total=int((num_files-start_file_count)/divide_by))
    while n_files*multiply_by < ((num_files/divide_by)-start_file_count):
        if progress_bar:
            n_files = (np.floor(len(os.listdir(filepath))/divide_by)*multiply_by)-start_file_count
            if n_files > last_count:
                loop.update(int(n_files-last_count))
                last_count = n_files
        time.sleep(1)
    if progress_bar:
        loop.update(int(((num_files/divide_by)-start_file_count)-last_count))