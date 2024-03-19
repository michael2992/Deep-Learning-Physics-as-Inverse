import os
import logging
import torch

torch.manual_seed(42)

# Define flags
flags = torch.zeros(1, dtype=torch.int)
flags.epochs = 10
flags.batch_size = 100
flags.save_dir = ""
flags.use_ckpt = False
flags.ckpt_dir = ""
flags.base_lr = 1e-3
flags.anneal_lr = True
flags.optimizer = "rmsprop"
flags.save_every_n_epochs = 5
flags.eval_every_n_epochs = 1
flags.print_interval = 10
flags.debug = False
flags.test_mode = False

# Setup logging
logger = logging.getLogger("torch")
logger.setLevel(logging.DEBUG)
# create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
