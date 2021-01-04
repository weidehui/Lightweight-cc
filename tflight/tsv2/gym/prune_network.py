import torch
import argparse
import numpy as np
from numpy import linalg as LA
from scipy.stats import rankdata
from collections import OrderedDict
import numpy as np
from numpy import linalg as LA
from scipy.stats import rankdata
from collections import OrderedDict
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prune_percentage', type=float, default=0.2)
    parser.add_argument('--logdir', type=str, default='log')
    args = parser.parse_known_args()[0]
    return args

def prune_layer(policy,model_path,args=get_args()):
    model = policy
    model.load_state_dict(torch.load(model_path))
    # Get all the weights
    weights = model.state_dict()
    # Get keys to access model weights
    layers = list(model.state_dict())
    #print("weights:   ",weights)
    #print("layers:  ",layers)
    ranks = {}
    pruned_weights = []
    counter=0
    for l in layers[1::2]:
        # Get weights for each layer and conver to numpy
        data = weights[l]
        counter+=2
        bias = weights[layers[counter]]
        #print("weight", l, ":  ", weights[l])
        w = data.cuda().data.cpu().numpy()
        # taking norm for each neuron
        norm = LA.norm(w, axis=0)
        # repeat the norm values to get the shape similar to that of layer weights
        #print("norm", l, ":  ", w.shape[0],"  ",w.shape[1])
        norm = np.tile(norm, (w.shape[0], 1))
        #print("norm_real: ",norm)
        # sort
        ranks[l] = (rankdata(norm, method='dense') - 1).astype(int).reshape(norm.shape)
        # Get the threshold value based on the value of k(prune percentage)
        lower_bound_rank = np.ceil(np.max(ranks[l]) * args.prune_percentage).astype(int)
        # Assign rank elements to 0 that are less than or equal to the threshold and 1 to those that are above.
        ranks[l][ranks[l] <= lower_bound_rank] = 0
        ranks[l][ranks[l] > lower_bound_rank] = 1
        # Multiply weights array with ranks to zero out the lower ranked weights
        w = w * ranks[l]
        # Assign the updated weights as tensor to data and append to the pruned_weights list
        data[...] = torch.from_numpy(w)
        pruned_weights.append(data)
    pruned_weights.append(weights[layers[-1]])
    new_state_dict = OrderedDict()
    torch.save(model.state_dict(),'parameter.pth')
    for l, pw in zip(layers, pruned_weights):
        new_state_dict[l] = pw
    model.state_dict = new_state_dict
    return model


