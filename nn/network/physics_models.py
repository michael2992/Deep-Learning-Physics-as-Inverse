import os
import shutil
import logging
import numpy as np
import torch
import torch.nn as nn
from pprint import pprint
import inspect
os.sys.path.append(".")
from nn.network.base import BaseNet, OPTIMIZERS
from nn.network.cells import BouncingODECell, SpringODECell, GravityODECell
from nn.network.encoder import ConvEncoder,ConvDecoder, VelEncoder
from nn.network import stn
#from nn.network.blocks import UNet, variable_from_network
from nn.utils.misc import log_metrics
from nn.utils.viz import gallery, gif
from nn.utils.math import sigmoid


import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.switch_backend('agg')

logger = logging.getLogger("torch")

CELLS = {
    "bouncing_ode_cell": BouncingODECell,
    "spring_ode_cell": SpringODECell,
    "gravity_ode_cell": GravityODECell,
    "lstm": nn.LSTMCell
}

COORD_UNITS = {
    "bouncing_balls": 8,
    "spring_color": 8,
    "spring_color_half": 8,
    "3bp_color": 12,
    "mnist_spring_color": 8
}

class PhysicsNet(BaseNet):
    def __init__(self,
                 task="",
                 recurrent_units=128,
                 lstm_layers=1,
                 cell_type="",
                 seq_len=20,
                 input_steps=3,
                 pred_steps=5,
                 autoencoder_loss=0.0,
                 alt_vel=False,
                 color=False,
                 input_size=36*36,
                 encoder_type="conv_encoder",
                 decoder_type="conv_st_decoder"):

        super(PhysicsNet, self).__init__()
        
        assert task in COORD_UNITS
        self.task = task
       
        self.recurrent_units = recurrent_units
        self.lstm_layers = lstm_layers

        self.cell_type = cell_type
        self.cell = CELLS[self.cell_type]
        self.color = color
        self.conv_ch = 3 if color else 1
        self.input_size = input_size

        self.conv_input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch]
        self.input_shape = [int(np.sqrt(input_size))]*2+[self.conv_ch]

        
        

        self.output_shape = self.input_shape

        assert seq_len > input_steps + pred_steps
        assert input_steps >= 1
        assert pred_steps >= 1
        self.seq_len = seq_len
        self.input_steps = input_steps
        self.pred_steps = pred_steps
        self.extrap_steps = self.seq_len-self.input_steps-self.pred_steps

        self.alt_vel = alt_vel
        self.autoencoder_loss = autoencoder_loss

        self.coord_units = COORD_UNITS[self.task]
        self.n_objs = self.coord_units//4

        self.extra_valid_fns.append((self.visualize_sequence, [], {}))
        self.extra_test_fns.append((self.visualize_sequence, [], {}))

        self.conv_encoder = ConvEncoder(self.input_shape, self.n_objs, self.conv_input_shape, self.conv_ch, self.alt_vel).forward
        self.vel_encoder = VelEncoder(self.input_shape, self.n_objs, self.conv_input_shape, self.coord_units, self.input_steps).forward  # Adjust parameters as needed
        self.conv_decoder = ConvDecoder(self.input_shape, self.n_objs, self.conv_input_shape, self.conv_ch, self.alt_vel).forward

        self.lstms = nn.LSTM(input_size=self.recurrent_units, hidden_size=self.recurrent_units, num_layers=self.lstm_layers)
        
        self.encoder = self.conv_encoder
        # {name: method for name, method in \
        #     inspect.getmembers(self, predicate=inspect.ismethod) if "encoder" in name
        # }[encoder_type]
        self.decoder = self.conv_decoder 
        # {name: method for name, method in \
        #     inspect.getmembers(self, predicate=inspect.ismethod) if "decoder" in name
        # }[decoder_type]  

    def get_batch(self, batch_size, iterator):
        batch_x, _ = iterator.next_batch(batch_size)
        batch_len = batch_x.shape[1]
        feed_dict = {'input': batch_x}  # Changed to PyTorch style
        return feed_dict, (batch_x, None)

    def compute_loss(self):
        recons_target = self.input[:, :self.input_steps+self.pred_steps]
        recons_loss = torch.square(recons_target - self.recons_out)
        recons_loss = torch.sum(recons_loss, dim=[2, 3, 4])

        self.recons_loss = torch.mean(recons_loss)

        target = self.input[:, self.input_steps:]
        loss = torch.square(target - self.output)
        loss = torch.sum(loss, dim=[2, 3, 4])

        self.pred_loss = torch.mean(loss[:, :self.pred_steps])
        self.extrap_loss = torch.mean(loss[:, self.pred_steps:])

        train_loss = self.pred_loss
        if self.autoencoder_loss > 0.0:
            train_loss += self.autoencoder_loss * self.recons_loss

        eval_losses = [self.pred_loss, self.extrap_loss, self.recons_loss]
        return train_loss, eval_losses

    def feedforward(self, x):
        x = torch.stack([torch.tensor(value, dtype=torch.float32) for value in x.values()], dim=0)
        self.input = x
        self.output = self.conv_feedforward(self.input)

        self.train_loss, self.eval_losses = self.compute_loss()
        self.train_metrics["train_loss"] = self.train_loss
        self.eval_metrics["eval_pred_loss"] = self.eval_losses[0]
        self.eval_metrics["eval_extrap_loss"] = self.eval_losses[1]
        self.eval_metrics["eval_recons_loss"] = self.eval_losses[2]
        self.loss = self.train_loss
        return self.output

    def build_optimizer(self, base_lr, optimizer="rmsprop", anneal_lr=True):
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.lr = base_lr  # No need to wrap this in nn.Parameter
        
        # Choose the optimizer
        if optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.lr)
        # Add other optimizers as needed
    
    # No need for manual gradient clipping setup here; do it in the training loop

    def conv_feedforward(self, input_tensor):
        # Initialize LSTM states
        h0 = torch.zeros(self.lstm_layers, input_tensor.size(0), self.recurrent_units).to(input_tensor.device)
        c0 = torch.zeros(self.lstm_layers, input_tensor.size(0), self.recurrent_units).to(input_tensor.device)
        
        # Reshape input tensor for encoding
        h = input_tensor[:, :self.input_steps+self.pred_steps].view(-1, *self.input_shape)
        enc_pos = self.encoder(h)  # Assuming self.encoder is correctly defined and returns encoded positions

        # Decode the input and pred frames
        recons_out = self.decoder(enc_pos)  # Assuming self.decoder is correctly defined
        self.recons_out = recons_out.view(input_tensor.size(0), self.input_steps+self.pred_steps, *self.input_shape)
        self.enc_pos = enc_pos.view(input_tensor.size(0), self.input_steps+self.pred_steps, self.coord_units//2)

        if self.input_steps > 1:
            vel = self.vel_encoder(self.enc_pos[:, :self.input_steps])  # Assuming self.vel_encoder is correctly defined
        else:
            vel = torch.zeros(input_tensor.size(0), self.coord_units//2).to(input_tensor.device)

        pos = self.enc_pos[:, self.input_steps-1]
        output_seq = []
        pos_vel_seq = [torch.cat([pos, vel], dim=1)]

        # Rollout ODE and decoder
        for t in range(self.pred_steps+self.extrap_steps):
            # Rollout
            pos, vel = self.rollout_cell.forward(pos, vel)  # Assuming rollout_cell is correctly defined

            # Decode
            out = self.decoder(pos)

            pos_vel_seq.append(torch.cat([pos, vel], dim=1))
            output_seq.append(out)

        output_seq = torch.stack(output_seq)

    def visualize_sequence(self):
        batch_size = 5

        feed_dict, (batch_x, _) = self.get_batch(batch_size, self.test_iterator)
        fetches = [self.output, self.recons_out]
        if hasattr(self, 'pos_vel_seq'):
            fetches.append(self.pos_vel_seq)

        output_seq, recons_seq, pos_vel_seq = self.feedforward(feed_dict, fetches)

        output_seq = np.concatenate([batch_x[:, :self.input_steps], output_seq], axis=1)
        recons_seq = np.concatenate([recons_seq, np.zeros((batch_size, self.extrap_steps) + recons_seq.shape[2:])], axis=1)

        for i in range(batch_x.shape[0]):
            to_concat = [output_seq[i], batch_x[i], recons_seq[i]]
            total_seq = np.concatenate(to_concat, axis=0)

            total_seq = total_seq.reshape([total_seq.shape[0], self.input_shape[0], self.input_shape[1], self.conv_ch])

            result = gallery(total_seq, ncols=batch_x.shape[1])

            norm = plt.Normalize(0.0, 1.0)

            figsize = (result.shape[1] // self.input_shape[1], result.shape[0] // self.input_shape[0])
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            fig.tight_layout()
            fig.savefig(os.path.join(self.save_dir, "example%d.jpg" % i))

        bordered_output_seq = 0.5 * np.ones([batch_size, self.seq_len, self.conv_input_shape[0] + 2,
                                            self.conv_input_shape[1] + 2, 3])
        bordered_batch_x = 0.5 * np.ones([batch_size, self.seq_len, self.conv_input_shape[0] + 2,
                                        self.conv_input_shape[1] + 2, 3])
        output_seq = output_seq.reshape([batch_size, self.seq_len] + self.input_shape)
        batch_x = batch_x.reshape([batch_size, self.seq_len] + self.input_shape)
        bordered_output_seq[:, :, 1:-1, 1:-1] = output_seq
        bordered_batch_x[:, :, 1:-1, 1:-1] = batch_x
        output_seq = bordered_output_seq
        batch_x = bordered_batch_x
        output_seq = np.concatenate(np.split(output_seq, batch_size, 0), axis=-2).squeeze()
        batch_x = np.concatenate(np.split(batch_x, batch_size, 0), axis=-2).squeeze()
        frames = np.concatenate([output_seq, batch_x], axis=1)

        gif(os.path.join(self.save_dir, "animation.gif"), frames * 255, fps=7, scale=3)

        fetches = {"contents": self.contents,
                "templates": self.template,
                "background_content": self.background_content,
                "transf_contents": self.transf_contents,
                "transf_masks": self.transf_masks,
                "enc_masks": self.enc_masks,
                "masked_objs": self.masked_objs}
        results = self.feedforward(feed_dict, fetches)
        np.savez_compressed(os.path.join(self.save_dir, "extra_outputs.npz"), **results)
        contents = results["contents"]
        templates = results["templates"]
        contents = 1 / (1 + np.exp(-contents))
        templates = 1 / (1 + np.exp(-(templates - 5)))
        if self.conv_ch == 1:
            contents = np.tile(contents, [1, 1, 1, 3])
        templates = np.tile(templates, [1, 1, 1, 3])
        total_seq = np.concatenate([contents, templates], axis=0)
        result = gallery(total_seq, ncols=self.n_objs)
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r, norm=norm)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "templates.jpg"))

        logger.info([(v.name, v.data) for v in self.parameters() if "ode_cell" in v.name or "sigma" in v.name])