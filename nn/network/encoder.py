import torch
import torch.nn as nn
import torch.nn.functional as F

#from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.network.blocks import UNet, VariableNetwork
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid
from nn.network.stn import SpatialTransformer

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
        print("Shape of x before transpose: {}".format(x.shape))
        x = torch.transpose(x, -1, 1)
        print("Shape of x after transpose: {}".format(x.shape))
        self.input_shape = x.shape
        print("self.conv_shape: {}".format(self.conv_input_shape))
        h = x
        if self.conv_input_shape[0] < 40:
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
        else:
            unet = UNet(h, 16, self.n_objs) # base_channels = 16 in original code but was 8 in our version?
            h = unet.forward(h)
            h = torch.cat([h, torch.ones_like(h[:,:1,:,:])], axis=1)
            h = torch.nn.functional.softmax(h, dim=1)

            self.enc_masks = h
            self.masked_objs = [self.enc_masks[:,i:i+1,:,:]*x for i in range(self.n_objs)]
            h = torch.cat(self.masked_objs, axis=0)

            # Pass through average pooling2d layer (not mentioned in the paper) and flatten
            h = torch.nn.functional.avg_pool2d(h, 2, 2)
            h = torch.flatten(h)

        # Pass through 2-layer location network
        location_net = LocationNetwork(h.shape[1])
        h = location_net.forward(h)
        print("h after location network: {}".format(h.shape))
        h = torch.cat(torch.split(h, self.n_objs, 0), axis=1)
        print("Shape of g after split and concat: {}".format(h.shape))

        # Pass through tanh activation layer to get output
        h = torch.tanh(h)*(self.conv_input_shape[0]/2) + (self.conv_input_shape[0]/2)
        print("Shape of h after encoding: {}".format(h[0].shape))

        if self.alt_vel:
            vels = self.vel_encoder(x)
            h = torch.cat([h, vels], axis=1)
            h = torch.nn.functional.relu(h)

        #cell = self.cell(self.recurrent_units)
        #c, h = cell(h)
        return h

class ConvDecoder(nn.Module):
    def __init__(self, inp, n_objs, conv_input_shape, conv_ch, alt_vel):
        super(ConvDecoder, self).__init__()
        self.inp = inp
        self.n_objs = n_objs
        self.conv_input_shape = conv_input_shape
        self.conv_ch = conv_ch
        self.alt_vel = alt_vel
        self.logsigma = torch.nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

         
    def forward(self, x):
        batch_size = x.shape[0]
        tmpl_size = self.conv_input_shape[0]//2
        log_tensor = torch.ones(tmpl_size, dtype=torch.float32)


        # Calculate sigma by taking the exponential of logsigma
        sigma = torch.exp(self.logsigma)
        vn_templ = VariableNetwork([self.n_objs, 1, tmpl_size, tmpl_size])
        template = vn_templ.forward(x)
        self.template = template
        template = torch.tile(template, [1,1,1,3])+5

        vn_cont = VariableNetwork([self.n_objs, self.conv_ch, tmpl_size, tmpl_size])
        contents = vn_cont.forward(x)
        self.contents = contents
        contents = torch.nn.Sigmoid(contents)
        joint = torch.cat([template, contents], axis=1)

        c2t = torch.from_numpy

        out_temp_cont = []

        for loc, join in zip(torch.split(x, self.n_objs, -1), torch.split(joint, self.n_objs, 0)):
            theta0 = torch.tile(c2t([sigma]), [torch.shape(x)[0]])
            theta1 =  torch.tile(c2t([0.0]), [torch.shape(x)[0]])
            theta2 = (self.conv_input_shape[0]/2-loc[:,0])/tmpl_size*sigma
            theta3 =  torch.tile(c2t([0.0]), [torch.shape(x)[0]])
            theta4 = torch.tile(c2t([sigma]), [torch.shape(x)[0]])
            theta5 = (self.conv_input_shape[0]/2-loc[:,1])/tmpl_size*sigma
            theta = torch.stack([theta0, theta1, theta2, theta3, theta4, theta5], axis=1)

            out_join = SpatialTransformer(torch.tile(join, [torch.shape(x)[0], 1, 1, 1]), theta, self.conv_input_shape[:2])
            out_temp_cont.append(torch.split(out_join, 1))
        
        background_content = VariableNetwork([1]+torch.shape(self.inp)).forward(x)
        self.background_content = torch.nn.Sigmoid(background_content)
        background_content = torch.tile(self.background_content, [batch_size, 1, 1,1])
        contents = [p[1] for p in out_temp_cont]
        contents.append(background_content)
        self.transf_contents = contents

        background_mask = torch.ones_like(out_temp_cont[0][0])
        masks = torch.stack([p[0]-5 for p in out_temp_cont]+[background_mask], axis=1)
        masks = torch.nn.Softmax(masks, axis=1)
        masks = torch.unbind(masks, axis=1)
        self.transf_masks = masks

        out = torch.sum([m*c for m,c in zip(masks, contents)])
        
        return out

class VelEncoder(nn.Module):
    """Estimates the velocity."""
    def __init__(self, input_channels, n_objs, input_shape, coord_units, input_steps):
        super(VelEncoder, self).__init__()
        self.input_channels = input_channels[0]
        self.n_objs = n_objs
        self.input_shape = input_shape
        self.coord_units = coord_units
        self.input_steps = input_steps

        self.layer1 = nn.Linear(self.input_channels, 100)
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, self.coord_units//self.n_objs//2)

    def forward(self, x):
        x = torch.split(x, self.n_objs, 2)
        x - torch.cat(x, dim=0)
        x = torch.reshape(x, [x.shape[0], self.input_steps*self.coord_units//self.n_objs//2])
        x = self.layer1(x)
        x  = self.tanh(x)
        x = self.layer2(x)
        x = self.tanh(x)
        output = self.output(x)
        return output

class LocationNetwork(nn.Module):
        """The 2-layer location network described in the paper.""" 
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