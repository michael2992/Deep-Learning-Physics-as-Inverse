import os
import sys
import shutil
import logging
import numpy as np
import torch  # Change import statement
import torch.nn as nn
from nn.utils.misc import log_metrics, zipdir

logger = logging.getLogger("torch")  # Change logger name

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "rmsprop": torch.optim.RMSprop,
    "momentum": lambda params: torch.optim.SGD(params, momentum=0.9),  # Change to use SGD with momentum
    "sgd": torch.optim.SGD
}


class BaseNet(nn.Module):

    def __init__(self):
        super(BaseNet, self).__init__()
        self.train_metrics = {}
        self.eval_metrics = {}

        self.extra_train_fns = []
        self.extra_valid_fns = []
        self.extra_test_fns = []

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Determine device

    def run_extra_fns(self, type):
        if type == "train":
            extra_fns = self.extra_train_fns
        elif type == "valid":
            extra_fns = self.extra_valid_fns
        else:
            extra_fns = self.extra_test_fns

        for fn, args, kwargs in extra_fns:
            fn(*args, **kwargs)

    def feedforward(self, inputs):
        raise NotImplementedError

    def compute_loss(self):
        raise NotImplementedError

    def build_graph(self):
        raise NotImplementedError

    def get_data(self, data_iterators):
        self.train_iterator, self.valid_iterator, self.test_iterator = data_iterators

    def get_iterator(self, type):
        if type == "train":
            eval_iterator = self.train_iterator 
        elif type == "valid":
            eval_iterator = self.valid_iterator 
        elif type == "test":
            eval_iterator = self.test_iterator
        return eval_iterator

    def initialize_graph(self,
                         save_dir, 
                         use_ckpt,
                         ckpt_dir=""):

        self.save_dir = save_dir
        if os.path.exists(save_dir):
            if use_ckpt:
                restore = True
                if ckpt_dir:
                    restore_dir = ckpt_dir
                else:
                    restore_dir = save_dir
            else:
                logger.info("Folder exists, deleting...")
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
                restore = False
        else:
            os.makedirs(save_dir)
            if use_ckpt:
                restore = True
                restore_dir = ckpt_dir 
            else:
                restore = False

        if restore:
            self.load_state_dict(torch.load(os.path.join(restore_dir, "model.pth")))  # Load model state
        else:
            self.to(self.device)  # Move model to device

    def build_optimizer(self, base_lr, optimizer="adam", anneal_lr=True):
        self.base_lr = base_lr
        self.anneal_lr = anneal_lr
        self.optimizer = OPTIMIZERS[optimizer](self.parameters(), lr=base_lr)  # Pass model parameters to optimizer

    def get_batch(self, batch_size, iterator):
        batch_x, batch_y = iterator.next_batch(batch_size)
        if batch_y is None:
            return torch.tensor(batch_x, device=self.device), None
        else:
            return torch.tensor(batch_x, device=self.device), torch.tensor(batch_y, device=self.device)

    def add_train_logger(self):
        log_path = os.path.join(self.save_dir, "log.txt")
        fh = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    def train(self,
              epochs, 
              batch_size,
              save_every_n_epochs,
              eval_every_n_epochs,
              print_interval,
              debug=False):

        self.add_train_logger()
        zipdir(root_path, self.save_dir) 
        logger.info("\n".join(sys.argv))

        step = 0

        # Run validation once before starting training
        if not debug and epochs > 0:
            valid_metrics_results = self.eval(batch_size, type='valid')
            log_metrics(logger, "valid - epoch=%s"%0, valid_metrics_results)

        for ep in range(1, epochs+1):
            if self.anneal_lr:
                if ep == int(0.75*epochs):
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] /= 5  # Reduce learning rate
            while self.train_iterator.epochs_completed < ep:
                inputs, targets = self.get_batch(batch_size, self.train_iterator)
                self.optimizer.zero_grad()
                outputs = self.feedforward(inputs)
                loss = self.compute_loss(outputs, targets)
                loss.backward()
                self.optimizer.step()

                self.run_extra_fns("train")

                if step % print_interval == 0:
                    # Convert loss to numpy for logging
                    log_metrics(logger, "train - iter=%s"%step, {'loss': loss.item()})
                step += 1

            if ep % eval_every_n_epochs == 0:
                valid_metrics_results = self.eval(batch_size, type='valid')
                log_metrics(logger, "valid - epoch=%s"%ep, valid_metrics_results)

            if ep % save_every_n_epochs == 0:
                torch.save(self.state_dict(), os.path.join(self.save_dir, "model.pth"))

        test_metrics_results = self.eval(batch_size, type='test')
        log_metrics(logger, "test - epoch=%s"%epochs, test_metrics_results)

    def eval(self,
             batch_size,
             type='valid'):

        eval_metrics_results = {k:[] for k in self.eval_metrics.keys()}
        eval_outputs = {"input":[], "output":[]}
        
        eval_iterator = self.get_iterator(type)
        eval_iterator.reset_epoch()
        
        while eval_iterator.get_epoch() < 1:
            if eval_iterator.X.shape[0] < 100:
                batch_size = eval_iterator.X.shape[0]
            inputs, targets = self.get_batch(batch_size, eval_iterator)
            #print("inputs for feedforward: {}".format(inputs.shape))
            outputs = self.feedforward(inputs)
            results = {k: metric(outputs, targets).item() for k, metric in self.eval_metrics.items()}

            for k in self.eval_metrics.keys():
                eval_metrics_results[k].append(results[k])
            eval_outputs["input"].append(inputs.cpu().numpy())  # Move tensors to CPU for numpy conversion
            eval_outputs["output"].append(outputs.cpu().numpy())

        eval_metrics_results = {k:np.mean(v, axis=0) for k,v in eval_metrics_results.items()}
        np.savez_compressed(os.path.join(self.save_dir, "outputs.npz"), 
                            input=np.concatenate(eval_outputs["input"], axis=0),
                            output=np.concatenate(eval_outputs["output"], axis=0))

        self.run_extra_fns(type)

        return eval_metrics_results
