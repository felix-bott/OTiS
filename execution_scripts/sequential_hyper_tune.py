import json
import numpy as np
import os 
from types import SimpleNamespace
import re
import glob
import sys

eval_criterion = "avg"

# Define the hyperparameter grid
batch_size_values = [32, 64, 128] # [16, 32, 64] #[16, 32, 64]
blr_values = [3e-6, 1e-5] # [3e-6, 1e-5, 3e-5, 1e-4]#[1e-6, 3e-6, 1e-5, 3e-5]
drop_path_values = [0.1] # [0.1, 0.0, 0.2] #[0.1, 0.0, 0.2] # default in Otis is 0.1, I guess for a reason.
layer_decay_values = [0.75] #[0.75, 0.5] #[0.75, 0.5]
weight_decay_values = [0.2] #[0.25, 0.05] #[0.25, 0.15, 0.05]

# Check for `trained_model_path` argument from the command line
if len(sys.argv) < 2:
    print("Usage: python sequential_hyper_tune.py /path/to/trained_model_directory")
    sys.exit(1)
trained_model_path = sys.argv[1]
# trained_model_path = "/vol/aimspace/users/bofe/TurBo/trainedModels/finetunedModels/OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5/Fold1"


def get_folder_names(directory):
    # List all folders in the directory, excluding those that contain COMPLETIONCERTIFICATE
    return [
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f))
        and glob.glob(os.path.join(directory, f, 'checkpoint*.pth'))
        and os.path.exists(os.path.join(directory, f, 'COMPLETIONCERTIFICATE.txt'))
    ]


def get_optim_setting(directory, eval_criterion):
    largest_value = None
    largest_folder = None

    for settingdir in os.listdir(directory):   
        if "COMPLETIONCERTIFICATE.txt" in os.listdir(os.path.join(directory, settingdir)):
            for outputfile in os.listdir(os.path.join(directory, settingdir)):
                if "checkpoint" in outputfile:
                    # Use regex to extract the value of the eval_criterion and convert to float
                    match = re.search(fr'{eval_criterion}-(.*?)\.pth', outputfile)
                    eval_criterion_value = float(match.group(1))

                    # If the current value is larger than the largest found so far
                    if largest_value is None or eval_criterion_value > largest_value:
                        largest_value = eval_criterion_value
                        largest_folder = settingdir
    
    numeric_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    optimal_values = re.findall(numeric_pattern, largest_folder)

    return [float(value) for value in optimal_values]


def iterate_settings(existing_settings, batch_size_values, blr_values, weight_decay_values, drop_path_values, layer_decay_values):
    # iterate over batch size and learning rate
    for layer_decay in layer_decay_values:
        for drop_path in drop_path_values:
            for weight_decay in weight_decay_values:
                for blr in blr_values:
                    for batch_size in batch_size_values:
                        curr_setting = [blr, batch_size, weight_decay, drop_path, layer_decay]
                        already_tested = curr_setting in existing_settings
                        if not already_tested:
                            print(f"{blr}:{batch_size}:{weight_decay}:{drop_path}:{layer_decay}")
                            return True


# determine which settings have already been tested
# Check if the folder exists, if not, create it
if not os.path.exists(trained_model_path):
    os.makedirs(trained_model_path)
    dir_list = []  # Return an empty list if the folder was just created
else:
    dir_list = get_folder_names(trained_model_path)
existing_settings = [s.split('_')[1:] for s in dir_list]
existing_settings = [[float(value) for value in sublist] for sublist in existing_settings]


# V1 =====================================================================
# # optimize batch_size and blr
# stop_script = iterate_settings(existing_settings, batch_size_values, blr_values, [weight_decay_values[0]], [drop_path_values[0]], [layer_decay_values[0]])
# if(stop_script):
#     exit()
# # select the best setting for batch_size and blr
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# batch_size_values_optim = [int(optimal_values[1])]
# blr_values_optim = [optimal_values[0]] 

# # optimize drop_path
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, [weight_decay_values[0]], drop_path_values, [layer_decay_values[0]])
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# drop_path_values_optim = [optimal_values[3]]

# # optimize layer_decay_values
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, [weight_decay_values[0]], drop_path_values_optim, layer_decay_values)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# layer_decay_values_optim = [optimal_values[4]]


# # optimize weight_decay
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, weight_decay_values, drop_path_values_optim, layer_decay_values_optim)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# weight_decay_values_optim = [optimal_values[2]] 


# V2 =====================================================================
# # optimize batch_size and blr and drop path
# stop_script = iterate_settings(existing_settings, batch_size_values, blr_values, [weight_decay_values[0]], drop_path_values, [layer_decay_values[0]])
# if(stop_script):
#     exit()
# # select the best setting for batch_size and blr
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# batch_size_values_optim = [int(optimal_values[1])]
# blr_values_optim = [optimal_values[0]] 
# drop_path_values_optim = [optimal_values[3]]

# # optimize layer_decay
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, [weight_decay_values[0]], drop_path_values_optim, layer_decay_values)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# layer_decay_values_optim = [optimal_values[4]]

# # optimize weight_decay
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, weight_decay_values, drop_path_values_optim, layer_decay_values_optim)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# weight_decay_values_optim = [optimal_values[2]] 

# # V3 =====================================================================
# # optimize batch_size and blr and layer_decay
# stop_script = iterate_settings(existing_settings, batch_size_values, blr_values, [weight_decay_values[0]], [drop_path_values[0]], layer_decay_values)
# if(stop_script):
#     exit()
# # select the best setting for batch_size and blr
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# batch_size_values_optim = [int(optimal_values[1])]
# blr_values_optim = [optimal_values[0]] 
# layer_decay_values_optim = [optimal_values[4]]

# # optimize drop path
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, [weight_decay_values[0]], drop_path_values, layer_decay_values_optim)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# drop_path_values_optim = [optimal_values[3]]

# # optimize weight_decay
# stop_script = iterate_settings(existing_settings, batch_size_values_optim, blr_values_optim, weight_decay_values, drop_path_values_optim, layer_decay_values_optim)
# if(stop_script):
#     exit()
# optimal_values = get_optim_setting(trained_model_path, eval_criterion)
# weight_decay_values_optim = [optimal_values[2]] 

# V4 =====================================================================
# optimize all jointly
stop_script = iterate_settings(existing_settings, batch_size_values, blr_values, weight_decay_values, drop_path_values, layer_decay_values)
if(stop_script):
    exit()




print("done")