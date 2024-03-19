import os
import logging
from nn.network import physics_models
from nn.utils.misc import classes_in_module
from nn.datasets.iterators import get_iterators
import runners.run_base

import torch
import torch.nn as nn

# Define flags
FLAGS = torch.zeros(1, dtype=torch.int)
FLAGS.task = ""
FLAGS.model = "PhysicsNet"
FLAGS.recurrent_units = 100
FLAGS.lstm_layers = 1
FLAGS.cell_type = ""
FLAGS.encoder_type = "conv_encoder"
FLAGS.decoder_type = "conv_st_decoder"
FLAGS.autoencoder_loss = 0.0
FLAGS.alt_vel = False
FLAGS.color = False
FLAGS.datapoints = 0

# Setup logging
logger = logging.getLogger("torch")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

model_classes = classes_in_module(physics_models)
Model = model_classes[FLAGS.model]

data_file, test_data_file, cell_type, seq_len, test_seq_len, input_steps, pred_steps, input_size = {
    "bouncing_balls": (
        "bouncing/color_bounce_vx8_vy8_sl12_r2.npz", 
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
}[FLAGS.task]

if __name__ == "__main__":
    if not FLAGS.test_mode:
        network = Model(FLAGS.task, FLAGS.recurrent_units, FLAGS.lstm_layers, cell_type, 
                        seq_len, input_steps, pred_steps,
                       FLAGS.autoencoder_loss, FLAGS.alt_vel, FLAGS.color, 
                       input_size, FLAGS.encoder_type, FLAGS.decoder_type)

        network.build_graph()
        network.build_optimizer(FLAGS.base_lr, FLAGS.optimizer, FLAGS.anneal_lr)
        network.initialize_graph(FLAGS.save_dir, FLAGS.use_ckpt, FLAGS.ckpt_dir)

        data_iterators = get_iterators(
                              os.path.join(
                                  os.path.dirname(os.path.realpath(__file__)), 
                                  "../data/datasets/%s"%data_file), conv=True, datapoints=FLAGS.datapoints)
        network.get_data(data_iterators)
        network.train(FLAGS.epochs, FLAGS.batch_size, FLAGS.save_every_n_epochs, FLAGS.eval_every_n_epochs,
                    FLAGS.print_interval, FLAGS.debug)
        
        torch.cuda.empty_cache()
    
    network = Model(FLAGS.task, FLAGS.recurrent_units, FLAGS.lstm_layers, cell_type, 
                    test_seq_len, input_steps, pred_steps,
                   FLAGS.autoencoder_loss, FLAGS.alt_vel, FLAGS.color, 
                   input_size, FLAGS.encoder_type, FLAGS.decoder_type)

    network.build_graph()
    network.build_optimizer(FLAGS.base_lr, FLAGS.optimizer, FLAGS.anneal_lr)
    network.initialize_graph(FLAGS.save_dir, True, FLAGS.ckpt_dir)

    data_iterators = get_iterators(
                          os.path.join(
                              os.path.dirname(os.path.realpath(__file__)), 
                              "../data/datasets/%s"%test_data_file), conv=True, datapoints=FLAGS.datapoints)
    network.get_data(data_iterators)
    network.train(0, FLAGS.batch_size, FLAGS.save_every_n_epochs, FLAGS.eval_every_n_epochs,
                FLAGS.print_interval, FLAGS.debug)
