"""
Global parameters in our code base
"""

import os
import torch

rootPath = "../data"   ## subject to change
savedPath = os.path.join(rootPath, 'model_param_saved')
pretrainedPath = os.path.join(rootPath, 'model_param_pretrained')

def buildIndexLabelMapping() :
    idx2label = os.listdir(os.path.join(rootPath, 'Moments_in_Time_Mini/jpg/validation'))
    label2idx = {}
    for i, label in enumerate(idx2label) :
        label2idx[label] = i
    return idx2label, label2idx

idx2label, label2idx = buildIndexLabelMapping()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
# device = torch.device("cpu")
