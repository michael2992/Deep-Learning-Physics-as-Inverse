import os

os.sys.path.append(".")
from nn.network import physics_models
from nn.utils.misc import classes_in_module
from nn.datasets.iterators import get_iterators

import torch
import torch.nn as nn

tasks = ["bouncing_balls", "spring_color", "spring_color_half", "3bp_color", "mnist_spring_color"]
task = tasks[0] # Use tasks[i] to choose a certain task, this should not be left blank
model = "PhysicsNet"
recurrent_units = 100
lstm_layers = 1
cell_type = ""
encoder_type = "conv_encoder"
decoder_type = "conv_st_decoder"
autoencoder_loss = 0.0
alt_vel = False
color = True
datapoints = 0

base_lr = 3e-4 
optimizer = "rmsprop" # Default optimizer is rmsprop, change name if you want to use a different optimizer
anneal_lr = True
save_dir = "runners\logs" # Replace this with an actual location
use_ckpt = False
ckpt_dir = "" # Replace with an actual location of a checkpoint if use_ckpt=true

epochs = 500
batch_size = 100
save_every_n_epochs = 5 # No default value given so idk what this should be
eval_every_n_epochs = 10
print_interval = 100
debug = False

model_classes = classes_in_module(physics_models)
Model = model_classes[model]
data_file, test_data_file, cell_type, seq_len, test_seq_len, input_steps, pred_steps, input_size = {"bouncing_balls": ( "bouncing/color_bounce_vx8_vy8_sl12_r2.npz", 
        "bouncing/color_bounce_vx8_vy8_sl30_r2.npz", 
        "bouncing_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color": (
        "spring_color/color_spring_vx8_vy8_sl12_r2_k4_e6.npz", 
        "spring_color/color_spring_vx8_vy8_sl30_r2_k4_e6.npz",
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "spring_color_half": (
        "spring_color_half/color_spring_vx4_vy4_sl12_r2_k4_e6_halfpane.npz", 
        "spring_color_half/color_spring_vx4_vy4_sl30_r2_k4_e6_halfpane.npz", 
        "spring_ode_cell",
        12, 30, 4, 6, 32*32),
    "3bp_color": (
        "3bp_color/color_3bp_vx2_vy2_sl20_r2_g60_m1_dt05.npz", 
        "3bp_color/color_3bp_vx2_vy2_sl40_r2_g60_m1_dt05.npz", 
        "gravity_ode_cell",
        20, 40, 4, 12, 36*36),
    "mnist_spring_color": (
        "mnist_spring_color/color_mnist_spring_vx8_vy8_sl12_r2_k2_e12.npz", 
        "mnist_spring_color/color_mnist_spring_vx8_vy8_sl30_r2_k2_e12.npz", 
        "spring_ode_cell",
        12, 30, 3, 7, 64*64)
}[task]

def main():
    network = Model(task, recurrent_units, lstm_layers, cell_type, 
                        seq_len, input_steps, pred_steps,
                       autoencoder_loss, alt_vel, color, 
                       input_size, encoder_type, decoder_type)
    
    network.build_optimizer(base_lr, optimizer, anneal_lr)
    network.initialize_graph(save_dir, use_ckpt, ckpt_dir)

    data_iterators = get_iterators(
                              os.path.join(
                                  os.path.dirname(os.path.realpath(__file__)), 
                                  "../data/datasets/%s"%data_file), conv=True, datapoints=datapoints)
    
    network.get_data(data_iterators)
    
    network.train(epochs, batch_size, save_every_n_epochs, eval_every_n_epochs,
                 print_interval, debug)
        
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()