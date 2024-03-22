import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid


class ConvEncoder(nn.Module):
    def __init__(self, input_channels, n_objs, conv_input_shape, conv_ch, alt_vel):
        super(ConvEncoder, self).__init__()
        # Initialize your layers here (if any)
        self.input_channels = input_channels
        self.n_objs = n_objs
        self.conv_input_shape = conv_input_shape
        self.conv_ch = conv_ch
        self.alt_vel = alt_vel
    def forward(self, x):
        rang = torch.range(self.conv_input_shape[0], self.conv_input_shape[1], dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(rang, rang)
        grid = torch.cat([grid_x[:,:,None], grid_y[:,:,None]], dim=2)
        grid = torch.tile(grid[None,:,:,:], [torch.shape(x)[0], 1, 1, 1])

        if self.input_shape[0] < 40:
            h = x
            h = shallow_unet(h, 8, self.n_objs, upsamp=True)

            h = torch.cat([h, torch.ones_like(h[:,:,:,:1])], axis=-1)
            h = torch.nn.functional.softmax(h, dim=-1)
            self.enc_masks = h
            self.masked_objs = [self.enc_masks[:,:,:,i:i+1]*x for i in range(self.n_objs)]

            h = torch.cat(self.masked_objs, axis=0)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0]*self.input_shape[1]*self.conv_ch])

        else:
            self.enc_masks = []
            self.masked_objs = []
            h = x
            for _ in range(4):
                h = unet(h, 8, self.n_objs, upsamp=True)
                h = torch.cat([h, torch.ones_like(h[:,:,:,:1])], axis=-1)
                h = torch.nn.functional.softmax(h, dim=-1)
                self.enc_masks.append(h)
                self.masked_objs.append([h[:,:,:,i:i+1]*x for i in range(self.n_objs)])

            h = torch.cat([torch.cat(mobjs, axis=0) for mobjs in self.masked_objs], axis=0)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0]*self.input_shape[1]*self.conv_ch])

        if self.alt_vel:
            vels = self.vel_encoder(x)
            h = torch.cat([h, vels], axis=1)
            h = torch.nn.functional.relu(h)

        cell = self.cell(self.recurrent_units)
        c, h = cell(h)
        return h

class ConvDecoder(nn.Module):
    def __init__(self, input_shape, n_objs, conv_input_shape, conv_ch, alt_vel):
        super(ConvDecoder, self).__init__()
        self.input_shape = input_shape
        self.n_objs = n_objs
        self.conv_input_shape = conv_input_shape
        self.conv_ch = conv_ch
        self.alt_vel = alt_vel
        
    def forward(self, x):
        if self.alt_vel:
            inp = inp[:,:-self.n_objs*2]
        h = x
        h = torch.nn.functional.relu(h)
        h = torch.nn.functional.relu(h)
        h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.conv_ch])

        if self.input_shape[0] < 40:
            h = shallow_unet(h, 8, self.n_objs, upsamp=True)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.n_objs])
            self.dec_masks = h
            self.dec_objs = [self.dec_masks[:,:,:,i:i+1]*inp for i in range(self.n_objs)]

            h = torch.cat(self.dec_objs, axis=3)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.conv_ch])

        else:
            self.dec_masks = []
            self.dec_objs = []
            for _ in range(4):
                h = unet(h, 8, self.n_objs, upsamp=True)
                h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.n_objs])
                self.dec_masks.append(h)
                self.dec_objs.append([h[:,:,:,i:i+1]*x for i in range(self.n_objs)])

            h = torch.cat([torch.cat(dobjs, axis=3) for dobjs in self.dec_objs], axis=3)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.conv_ch])
        return h

class VelEncoder(nn.Module):
    def __init__(self, input_channels, n_objs, input_shape):
        super(VelEncoder, self).__init__()
        self.input_channels = input_channels
        self.n_objs = n_objs
        self.input_shape = input_shape

    def forward(self, x):
        h = x
        h = shallow_unet(h, 8, self.n_objs*2, upsamp=True)
        h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0]*self.input_shape[1]*self.n_objs*2])
        return h 