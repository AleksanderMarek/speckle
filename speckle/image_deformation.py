import os
from numpy import genfromtxt
import numpy as np
"""
Created on Sat Jan 14 22:46:05 2023

This function performs image deformation on an image, then 2D correlation and
returns the mean and noise floor of the result

@author: Alex Marek
"""

def image_deformation(im_path, imdef_inp, corr_inp):
    # Modify imdef_inp file and run image deformation
    search_str = "<Reference$image>"
    valid_str = f"<Reference$image>=<{im_path}>\n"
    modify_MatchID_input(imdef_inp, search_str, valid_str)
    shell_cmd = \
    f'\"\"C:\Program Files (x86)\MatchID\MatchID 2D\MatchID.exe\" ' +\
    f'\"{imdef_inp}\"\"'
    os.system(shell_cmd)
    # Run correlation
    shell_cmd = \
    f'\"\"C:\Program Files (x86)\MatchID\MatchID 2D\MatchID.exe\" ' +\
    f'\"{corr_inp}\"\"'
    os.system(shell_cmd)
    # Import the results
    target_U = 0.382;
    results_path = r"D:\Experiment Quality\ImDef\im_deformed_1_0.def.csv"
    results = genfromtxt(results_path, delimiter=',')
    if len(results.shape) == 2:
        U = results[1:,2]
        U_corrected = np.nan_to_num(U)
        mean_U = np.mean(U_corrected)
        noise_floor = \
            np.linalg.norm(U_corrected - target_U) / U_corrected.shape[0]
    else:
        mean_U = 0.0
        noise_floor = target_U        
    return noise_floor, mean_U
    
def modify_MatchID_input(file_path, search_str, replace_str):
    with open(file_path, 'r') as file:
        # read a list of lines into data
        data = file.readlines()
    for ii in range(len(data)):
        if search_str in data[ii]:
            data[ii] = replace_str
            break
    with open(file_path, 'w') as file:
        file.writelines( data )        
