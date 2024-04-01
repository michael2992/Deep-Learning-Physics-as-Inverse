import numpy as np
import torch

""" Useful subnetwork components """


def unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h1 = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.MaxPool2d(h1, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h2 = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.MaxPool2d(h2, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    h3 = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.MaxPool2d(h3, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*8, 3, activation=torch.nn.ReLU(), padding="SAME")
    h4 = torch.nn.Conv2d(h, base_channels*8, 3, activation=torch.nn.ReLU(), padding="SAME")
    if upsamp:
        h = torch.nn.functional.interpolate(h4, size=h3.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*4, 3, 2, activation=None, padding="SAME")
    h = torch.cat([h, h3], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h2.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="SAME")
    h = torch.cat([h, h2], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h1.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, activation=None, padding="SAME")
    h = torch.cat([h, h1], dim=-1)
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")

    h = torch.nn.Conv2d(h, out_channels, 1, activation=None, padding="SAME")
    return h


def shallow_unet(inp, base_channels, out_channels, upsamp=True):
    h = inp
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h1 = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.MaxPool2d(h1, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h2 = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.MaxPool2d(h2, 2, 2)
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels*4, 3, activation=torch.nn.ReLU(), padding="SAME")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h2.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels*2, 3, 2, activation=None, padding="SAME")
    h = torch.cat([h, h2], dim=-1)
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels*2, 3, activation=torch.nn.ReLU(), padding="SAME")
    if upsamp:
        h = torch.nn.functional.interpolate(h, size=h1.shape[2:])
        h = torch.nn.Conv2d(h, base_channels*2, 3, activation=None, padding="SAME")
    else:
        h = torch.nn.ConvTranspose2d(h, base_channels, 3, 2, activation=None, padding="SAME")
    h = torch.cat([h, h1], dim=-1)
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")
    h = torch.nn.Conv2d(h, base_channels, 3, activation=torch.nn.ReLU(), padding="SAME")

    h = torch.nn.Conv2d(h, out_channels, 1, activation=None, padding="SAME")
    return h


def variable_from_network(shape):
    # Produces a variable from a vector of 1's. 
    # Improves learning speed of contents and masks.
    var = torch.ones([1,10])
    var = torch.nn.Linear(var, 200, activation=torch.nn.Tanh())
    var = torch.nn.Linear(var, np.prod(shape), activation=None)
    var = var.view(shape)
    return var
