import numpy as np
import os
from pathlib import Path
import pdb

def load_data_from_path(path, subject, start_epoch, channels):
    DATA_LIST = []
    for channel in channels:
        npz = np.load(os.path.join(path, channel, 
            '{}_{}.npy'.format(subject, start_epoch)))
        DATA_LIST.append(npz.flatten())

    return DATA_LIST


def load_grad_pred(path_list, target_model, subject, start_epoch):
    path_grad = Path(path_list[0]).joinpath(
        f'{target_model}/{subject}_{start_epoch}.npy')
    path_pred = Path(path_list[1]).joinpath(
        f'{target_model}/{subject}_{start_epoch}.npy')

    if path_grad.exists():
        grad = np.load(path_grad)
    else:
        print('No gradient file')
        grad = None

    if path_pred.exists():
        pred = np.load(path_pred)
    else:
        print('No prediction file')
        pred = None

    return (grad, pred)


def load_labels(path_list, subject, start_epoch):
    path_label1 = Path(path_list[0]).joinpath(
        f'{subject}_{start_epoch}.npy')
    path_label2 = Path(path_list[1]).joinpath(
        f'{subject}_{start_epoch}.npy')

    if path_label1.exists():
        label1 = np.load(path_label1)
    else:
        print('No label1 file')
        label1 = None

    if path_label2.exists():
        label2 = np.load(path_label2)
    else:
        print('No label2 file')
        label2 = None

    return (label1, label2)


def load_activation(path, subject, start_epoch):
    path_act = Path(path).joinpath(
        f'{subject}_{start_epoch}.npz')

    if path_act.exists():
        act = np.load(path_act)
    else:
        print('No activation file')
        act = None

    return act