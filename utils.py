import os
import shutil

import torch

def init_weights(m):
    """
    All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def save_checkpoints(state, checkpoint, is_G):
    """ Saves model and training parameters at checkpoint + 'last.pth.
    Args:
        state : (dict) contains model's state_dict
        checkpoint : (string) folder where parameters are to be saved
        is_G : (bool) if Generator, then True
    """
    if is_G:
        filepath = os.path.join(checkpoint, 'last_G.pth')
    else:
        filepath = os.path.join(checkpoint, 'last_D.pth')
    if not os.path.exists(checkpoint):
        print("Checkpoint Direcotry doesn't exist. Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        pass
    torch.save(state, filepath)