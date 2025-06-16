import os
import time
import numpy as np
import tqdm

def mp_run_fit(args: list) -> None:

    """
    Run the reinforcement learning fit in a multiprocessing context.

    Parameters
    ----------
    args : list
        A list where the first element is the rl_pipeline object and the rest are parameters for the fit method.
    """

    rl_pipeline = args[0]
    rl_pipeline.run_rl_fit(args[1:])

def mp_run_simulations(args: list) -> None:

    """
    Run the reinforcement learning simulations in a multiprocessing context.

    Parameters
    ----------
    args : list
        A list where the first element is the rl_pipeline object and the rest are parameters for the run_simulations method.
    """

    rl_pipeline = args[0]
    if 'generate_data' in str(args[-1]):
        rl_pipeline.run_simulations(args[1:-1], generate_data=args[-1].split('=')[-1])
    else:
        rl_pipeline.run_simulations(args[1:])

def mp_progress(num_files: int, filepath: str = 'RL/fits/temp', divide_by: int = 1, multiply_by: int = 1, progress_bar: bool = True) -> None:

    """
    Monitor the progress of file generation in a specified directory.
    
    Parameters
    ----------
    num_files : int
        The total number of files expected to be generated.
    filepath : str, optional
        The directory where the files are being generated. Default is 'RL/fits/temp'.
    divide_by : int, optional
        The factor by which to divide the number of files for progress tracking. Default is 1.
    multiply_by : int, optional
        The factor by which to multiply the number of files for progress tracking. Default is 1.
    progress_bar : bool, optional
        Whether to display a progress bar. Default is True.
        
    Returns
    -------
    None
    """

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