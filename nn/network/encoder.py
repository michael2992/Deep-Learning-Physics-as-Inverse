import torch
import torch.nn as nn
import torch.nn.functional as F

#from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.network.blocks import UNet, variable_from_network
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
        #print("Conv input shape: {}".format(self.conv_input_shape))
        self.conv_ch = conv_ch
        self.alt_vel = alt_vel

    def forward(self, x):
        #rang = torch.range(self.conv_input_shape[0], self.conv_input_shape[1], dtype=torch.float32)
        #grid_x, grid_y = torch.meshgrid(rang, rang)
        #grid = torch.cat([grid_x[:,:,None], grid_y[:,:,None]], dim=2)
        #grid = torch.tile(grid[None,:,:,:], [x.shape[0], 1, 1, 1])
        print("Shape of x before transpose: {}".format(x.shape))
        x = torch.transpose(x, -1, 1)
        print("Shape of x after transpose: {}".format(x.shape))
        self.input_shape = x.shape
        print("self.conv_shape: {}".format(self.conv_input_shape))
        if self.conv_input_shape[0] < 40:
            h = x
            #h = shallow_unet(h, 8, self.n_objs, upsamp=True)
            shallow_unet = UNet(h, 8, self.n_objs, depth=3)
            h = shallow_unet.forward(h)

            # Add learnable bg mask
            print("Shape of h before adding learnable bg mask: {}".format(h.shape)) # (batch, channels, h, w)
            h = torch.cat([h, torch.ones_like(h[:,:1,:,:])], axis=1)
            print("Shape of h after adding learnable bg mask: {}".format(h.shape)) # Should be (batch, channels+1, h, w) 

            # Pass through softmax
            h = torch.nn.functional.softmax(h, dim=1)

            # Multiply input image with each mask
            self.enc_masks = h
            self.masked_objs = [self.enc_masks[:,i:i+1,:,:]*x for i in range(self.n_objs)]
            h = torch.cat(self.masked_objs, axis=0)

            # Produce x,y-coordinates (this part appears to be different from the paper description)
            print("Shape of h before reshaping: {}".format(h.shape))
            h = torch.reshape(h, [h.shape[0], self.conv_input_shape[0]*self.conv_input_shape[0]*self.conv_ch]) 
            print("Shape of h after reshaping: {}".format(h.shape))
            mask_net = MaskNetwork(h.shape[1])
            h = mask_net.forward(h)
            print("h after location network: {}".format(h.shape))
            h = torch.cat(torch.split(h, self.n_objs, 0), axis=1)
            print("Shape of g after split and concat: {}".format(h.shape))

            # Pass through tanh activation layer to get output
            h = torch.tanh(h)*(self.conv_input_shape[0]/2) + (self.conv_input_shape[0]/2)
            print("Shape of h after encoding: {}".format(h[0].shape))

        else:
            self.enc_masks = []
            self.masked_objs = []
            h = x
            for _ in range(4):
                #h = unet(h, 8, self.n_objs, upsamp=True)
                unet = UNet(h, 16, self.n_objs) # base_channels = 16 in original code but 8 in the line above?
                h = unet.forward(h)
                h = torch.cat([h, torch.ones_like(h[:,:1,:,:])], axis=1)
                h = torch.nn.functional.softmax(h, dim=1)
                self.enc_masks.append(h)
                self.masked_objs.append([h[:,i:i+1,:,:]*x for i in range(self.n_objs)])

            h = torch.cat([torch.cat(mobjs, axis=0) for mobjs in self.masked_objs], axis=0)
            h = torch.reshape(h, [h.shape[0], self.conv_input_shape[0]*self.conv_input_shape[0]*self.conv_ch])

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
            #h = shallow_unet(h, 8, self.n_objs, upsamp=True)
            shallow_unet = UNet(h, 8, self.n_objs, depth=3)
            h = shallow_unet.forward(h)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.n_objs])
            self.dec_masks = h
            self.dec_objs = [self.dec_masks[:,:,:,i:i+1]*inp for i in range(self.n_objs)]

            h = torch.cat(self.dec_objs, axis=3)
            h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0], self.input_shape[1], self.conv_ch])

        else:
            self.dec_masks = []
            self.dec_objs = []
            for _ in range(4):
                #h = unet(h, 8, self.n_objs, upsamp=True)
                unet = UNet(h, 8, self.n_objs)
                h = unet.forward(h)
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
        #h = shallow_unet(h, 8, self.n_objs*2, upsamp=True)
        shallow_unet = UNet(h, 8, self.n_objs*2, depth=3)
        h = shallow_unet.forward(h)
        h = torch.reshape(h, [torch.shape(h)[0], self.input_shape[0]*self.input_shape[1]*self.n_objs*2])
        return h 
    
class MaskNetwork(nn.Module):
        """The 2-layer location network described in the paper (even though it has 3 layers??)."""
        def __init__(self, input):
            super().__init__()
            self.layer1 = nn.Linear(input, 200)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(200, 200)
            self.output = nn.Linear(200, 2)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            output = self.output(x)
            return output