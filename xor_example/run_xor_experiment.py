import tensorflow as tf
import numpy as np
import glob
import datetime

from experiment import Experiment
from models.tf_mlp_model import TF_MLP_Model
from trainers.tf_trainer import TF_Trainer
from evaluators.evaluator import Evaluator

from trainers.tf_utils.losses import TF_MSE_Loss
from trainers.tf_utils.optimizers import TF_Adam_Optimizer

from xor_dataset_utils import create_xor_dataset, XOR_Example_Parser

def setup_experiment():
    # setup modules & hyperparameters
    XOR_experiment = Experiment()
    if True:
        XOR_model = TF_MLP_Model()
        if True:
            XOR_model.in_size = 2
            XOR_model.hidden_sizes = [20, 20, 20, 20]
            XOR_model.out_size = 1

            XOR_model.activation = "relu"

        XOR_trainer = TF_Trainer()
        if True:
            XOR_optimizer = TF_Adam_Optimizer()
            if True:
                XOR_optimizer.learning_rate = 0.01
                XOR_optimizer.epsilon = 0.1

            XOR_loss = TF_MSE_Loss()

            XOR_example_parser = XOR_Example_Parser()

            XOR_trainer.random_seed = 0

            XOR_trainer.data_prefix = "data/xor"
            XOR_trainer.example_parser = XOR_Example_Parser()
            XOR_trainer.dataset_shuffle_buffer_size = 4

            XOR_trainer.loss = TF_MSE_Loss()
            XOR_trainer.optimizer = XOR_optimizer

            XOR_trainer.load_checkpoint_dir = None
            XOR_trainer.start_epoch = 0

            XOR_trainer.n_epochs = 50
            XOR_trainer.batch_size = 4
            XOR_trainer.log_period = 1
            XOR_trainer.save_period = 50

        XOR_evaluator = Evaluator()

        XOR_experiment.model = XOR_model
        XOR_experiment.trainer = XOR_trainer
        XOR_experiment.evaluator = XOR_evaluator

    return XOR_experiment

if __name__ == "__main__":
    create_xor_dataset()

    # setup experiment for the first time
    XOR_experiment = setup_experiment()
    # restore experiment from a params file
    #XOR_experiment = Experiment()
    #XOR_experiment.load("last_params")

    XOR_experiment.set_exp_name("xor")

    XOR_experiment.train()

    XOR_experiment.save()
