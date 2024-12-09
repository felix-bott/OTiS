import torch
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.feature_selection import r_regression

set_type = 'test'
labels_ind = 0
matplotlib.use('Agg')

# Initialize empty lists to store concatenated results across folds
all_labels_avg = []
all_logits_avg = []

# Initialize dictionaries to store results for each fold
pcc_avg = {}
r_squared_avg = {}

for currFold in ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']:
    # path to predictions
    predictions_path = f"/vol/aimspace/users/bofe/TurBo/trainedModels/noPretrainingCVjointTune2/OrthoAggSchaefer100_freq-all_epochLength-20_overlap-0.5/evalResults/{currFold}/logits/"


    logits_file = predictions_path + 'logits_'+set_type+'.pt'
    labels_file = predictions_path + 'labels_'+set_type+'.pt'
    subjects_file = predictions_path + 'subjectList_'+set_type+'.csv'

    # Load labels and logits
    labels = torch.load(labels_file) 
    logits = torch.load(logits_file)
    labels = labels.numpy()
    logits = logits.numpy()

    # Recompute some metrics (on EEG-epoch-level data) to check consistency with wandb ouput
    corr_mat = np.corrcoef(labels[:,labels_ind], logits[:,labels_ind])
    pcc = corr_mat[0, 1]
    r_squared = r2_score(labels[:,labels_ind], logits[:,labels_ind])

    #plt.scatter(labels, logits, alpha=0.2, color='blue')

    # Load the subjects list
    subjects = np.genfromtxt(subjects_file, delimiter=',', dtype=str)  # Assuming each row is a subject ID

    uniqueSubjects = np.unique(subjects)

    # Accumulate predictions and labels by subject
    labels_avg = np.full((len(uniqueSubjects),1), np.nan)
    logits_avg = np.full((len(uniqueSubjects),1), np.nan)
    for i, currSubject in enumerate(uniqueSubjects):
        labels_avg[i] = np.mean(labels[subjects == currSubject, labels_ind])
        logits_avg[i] = np.mean(logits[subjects == currSubject, labels_ind])


    all_labels_avg.append(labels_avg)
    all_logits_avg.append(logits_avg)

    # Compute some metrics (on subject-level data) to check consistency with wandb ouput
    corr_mat_avg = np.corrcoef(labels_avg.flatten(), logits_avg.flatten())
    pcc_avg[currFold] = corr_mat_avg[0, 1]
    r_squared_avg[currFold] = r2_score(labels_avg.flatten(), logits_avg.flatten())

    print(f"pcc_avg: {pcc_avg[currFold]:.3f}; r2_avg: {r_squared_avg[currFold]:.3f}")

    # plot labels against logits
    plt.figure(figsize=(10, 10))
    plt.scatter(labels_avg, logits_avg, alpha=0.9, color='red')
    # plt.plot([min(labels), max(labels)], [min(labels), max(labels)], color='red', linestyle='--')  # Identity line
    plt.xlabel('Observations')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.title('observations vs. predictions')
    plt.grid(True)

    plt.savefig(predictions_path+'visualization'+set_type+'.png')
    plt.close()

all_labels_avg = np.vstack(all_labels_avg)
all_logits_avg = np.vstack(all_logits_avg)

corr_mat_avg = np.corrcoef(all_labels_avg.flatten(), all_logits_avg.flatten())
all_pcc_avg = corr_mat_avg[0, 1]
all_r_squared_avg = r2_score(all_labels_avg.flatten(), all_logits_avg.flatten())
print(f"all_pcc_avg: {all_pcc_avg:.3f}; all_r2_avg: {all_r_squared_avg:.3f}")