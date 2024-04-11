import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from nn.network.blocks import unet, shallow_unet, variable_from_network
from nn.network.blocks import UNet, VariableNetwork
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid
from nn.network.stn import SpatialTransformer
import matplotlib.pyplot as plt
import numpy as np
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
        # #print("Shape of x before transpose: {}".format(x.shape))
        # x = torch.transpose(x, -1, 1)
        # #print("Shape of x after transpose: {}".format(x.shape))
      
        self.input_shape = x.shape
        #print("self.conv_shape: {}".format(self.conv_input_shape))
        h = x
        if self.conv_input_shape[0] < 40:
            shallow_unet = UNet(h, 8, self.n_objs, depth=3)
            h_out = []
            for t in range(h.size(-1)):
                current_h = h[:,:,:,:,t]
                next_h = shallow_unet.forward(current_h)
                h_out.append(next_h)
            
            h = torch.stack(h_out, dim=-1)

            # Add learnable bg mask
            #print("Shape of h before adding learnable bg mask: {}".format(h.shape)) # (batch, channels, h, w)
            h = torch.cat([h, torch.ones_like(h[:,:1,:,:])], axis=1)
            #print("Shape of h after adding learnable bg mask: {}".format(h.shape)) # Should be (batch, channels+1, h, w) 

            # Pass through softmax

            h = torch.nn.functional.softmax(h, dim=1)
            #print("Shape after softmax: {}".format(h.shape))
            # Multiply input image with each mask
            self.enc_masks = h
            self.masked_objs = [self.enc_masks[:,i:i+1,:,:]*x for i in range(self.n_objs)]
            #print("Shape after masked objs: {}".format(len(self.masked_objs)))
            h = torch.cat(self.masked_objs, axis=1)
            #print("Shape after cat masked: {}".format(h.shape))
            # Produce x,y-coordinates (this part appears to be different from the paper description)
            #print("Shape of h before reshaping: {}".format(h.shape))
            # #print([h.shape[0], self.conv_input_shape[1]*self.conv_input_shape[2]*self.conv_ch])
            # h = torch.reshape(h, [h.shape[0], self.conv_input_shape[1]*self.conv_input_shape[2]*self.conv_ch,h.shape[-1] ]) 
            # #print("Shape of h after reshaping: {}".format(h.shape)) 
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
        #print("Before location network",h.shape)
        # Pass through 2-layer location network
        location_net = LocationNetwork(input=6*32*32, n_objs=self.n_objs)  # Adjust 'input' based on the flattening of [6, 32, 32]

        output_h = []

        for batch_idx in range(h.shape[0]):  # Looping over batch
            temp_outputs = []
            for temp_idx in range(h.shape[-1]):  # Looping over temporal steps
                # Selecting the [6, 32, 32] image for the current batch and temporal index
                img = h[batch_idx, :, :, :, temp_idx]
                
                # Flatten the img to match LocationNetwork's input shape
                img_flat = img.reshape(-1)  # Flatten the image
                img_flat = img_flat.unsqueeze(0)  # Adding a batch dimension for compatibility with nn.Module
                
                # Apply the location network
                loc_output = location_net(img_flat).reshape(-1,self.n_objs, 2)
                
                # Store the output
                temp_outputs.append(loc_output)

            # Combine temporal outputs for the current batch item
            output_h.append(torch.stack(temp_outputs, dim=0))

        # Combine all batch outputs to form the new 'h'
        h = torch.stack(output_h, dim=0)
        #print("h after location network: {}".format(h.shape))

        
        # h = location_net.forward(h)
        #print("h after location network: {}".format(h.shape))
        h = torch.cat(torch.split(h, self.n_objs, 1), dim=1)
        #print("Shape of g after split and concat: {}".format(h.shape))

        # Pass through tanh activation layer to get output
        h = torch.tanh(h)*(self.conv_input_shape[0]/2) + (self.conv_input_shape[0]/2)
        #print("Shape of h after encoding: {}".format(h.shape))
        if self.alt_vel:
            vels = self.vel_encoder(x)
            h = torch.cat([h, vels], axis=1)
            h = torch.nn.functional.relu(h)

        #cell = self.cell(self.recurrent_units)
        #c, h = cell(h)
        # h_reshaped = h.view(360, 2, 2)
        return h.squeeze(2)

class ConvDecoder(nn.Module):
    def __init__(self, inp, n_objs, conv_input_shape, conv_ch, alt_vel=False):
        super(ConvDecoder, self).__init__()
        self.inp = list(inp)
        self.n_objs = n_objs
        self.conv_input_shape = conv_input_shape
        self.conv_ch = conv_ch
        self.alt_vel = alt_vel
        self.tmpl_size = self.conv_input_shape[1] // 2
        self.logsigma = torch.nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

         # Initialize your VariableNetworks here, assuming they are defined elsewhere
        self.vn_templ = VariableNetwork([self.n_objs,  self.conv_ch, self.tmpl_size,self.tmpl_size])
        self.vn_cont = VariableNetwork([self.n_objs, 1, self.tmpl_size, self.tmpl_size])
        self.stn = SpatialTransformer(self.conv_input_shape[1:])
        self.vn_background = VariableNetwork([1, *self.conv_input_shape])  # Adjusted based on assumed input shape

         
    def forward(self, x):
        #print("\n======================\nDecoder\n")
        #print("X_shape", x.shape)
        print("Shape of x in decoder ", x.shape)
        batch_size, temporal, n_objs, coordinates = x.shape  # Updated shape
        #coordinates, n_objs = x.shape
        tmpl_size = self.tmpl_size

        sigma = torch.exp(self.logsigma)
        nill = torch.zeros_like(sigma)

        template = self.vn_templ.forward(x)
        self.template = template
        template = torch.tile(template, [1,3,1,1])+5

        contents = torch.sigmoid(self.vn_cont.forward(x))
        self.contents = contents


        #print(template.shape, contents.shape)
        joint = torch.cat([template, contents], axis=1)
        #print("join_shape", joint.shape)
        
        out_temp_cont = []
        theta_lst = []
        
        # for loc, join in zip(torch.split(x,self.n_objs, 1), torch.split(joint, self.n_objs, 0)):
        join = torch.split(joint, self.n_objs, 0)[0]
        U = torch.tile(join, [1, 1,1,1])

        output_size = [2, temporal, self.conv_input_shape[1],self.conv_input_shape[1]]

        for b_idx in range(batch_size):
            for temp_idx in range(temporal):
                theta_img = []
                for obj_idx in range(n_objs):
                    theta2 = (self.conv_input_shape[0]/2-x[b_idx, temp_idx, obj_idx,0])/tmpl_size*sigma
                    theta5 = (self.conv_input_shape[0]/2-x[b_idx, temp_idx, obj_idx,1])/tmpl_size*sigma

                    theta = torch.stack([sigma, nill, theta2, nill, sigma, theta5], axis=0).view(2,3)
                               
                    theta_img.append(theta)

                out_join = self.stn.forward(U, torch.stack(theta_img))
                ##print("out_join", out_join.shape)
                out_temp_cont.append(out_join)
                
        out_temp_cont = torch.stack(out_temp_cont, dim=0)  # Stack along the temporal dimension

        ##print("out_temp shape",out_temp_cont.shape)
        
        self.background_content = torch.nn.functional.sigmoid(self.vn_background.forward(x))
        background_content = torch.tile(self.background_content, [batch_size, 1, 1, 1])
        contents = [p[1] for p in out_temp_cont]
        contents.append(background_content)
        self.transf_contents = contents

        
        background_mask = torch.ones_like(out_temp_cont[0][0])
        masks = torch.stack([p[0]-5 for p in out_temp_cont], axis=-1)
        masks = torch.nn.functional.softmax(masks, dim=-1)
        masks = torch.unbind(masks, axis=-1)
        self.transf_masks = masks


        # #print(len(masks), len(masks[0]), len(masks[0][0]), len(masks[0][0][0]) )
        # out = torch.sum(torch.stack([m * c for m, c in zip(masks, contents)]),dim=0)
        mult_lst = []
        for m, c in zip(masks, contents):
            mult = m * c  # Element-wise multiplication
            mult_lst.append(mult)

        # Convert list of tensors to a single tensor before summing
        out = torch.sum(torch.stack(mult_lst), dim=1)
                
        ##print("decoder_out", out.shape)
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

        self.layer1 = nn.Linear(self.input_steps*self.coord_units//self.n_objs//2, 100) 
        self.tanh = nn.Tanh()
        self.layer2 = nn.Linear(100, 100)
        self.output = nn.Linear(100, self.coord_units//self.n_objs//2)

    def forward(self, x):
        x = torch.split(x, self.n_objs, 2)
        print("Shape after split: ", len(x))
        x = torch.cat(x, dim=0)
        x = torch.reshape(x, [x.shape[0], self.input_steps*self.coord_units//self.n_objs//2])
        print("Shape of x in velencoder before going through layer 1, ", x.shape)
        x = self.layer1(x)
        x  = self.tanh(x)
        x = self.layer2(x)
        x = self.tanh(x)
        output = self.output(x)
        return output

class LocationNetwork(nn.Module):
        """The 2-layer location network described in the paper.""" 
        def __init__(self, input, n_objs):
            super().__init__()
            self.layer1 = nn.Linear(input, 200)
            self.relu = nn.ReLU()
            self.layer2 = nn.Linear(200, 200)
            self.output = nn.Linear(200, 2*n_objs)

        def forward(self, x):
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            output = self.output(x)
            return output
        

        