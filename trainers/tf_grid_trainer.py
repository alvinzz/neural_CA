from trainers.trainer import Trainer

import tensorflow as tf
import numpy as np
import glob
import tqdm

class TF_Grid_Trainer(Trainer):
    param_path = "trainers.tf_grid_trainer"
    param_name = "TF_Grid_Trainer"

    def __init__(self):
        self.global_params = [
            "train_time_horizon"
        ]

        self.params = [
            "loss",
            "optimizer",
            
            "batch_size",

            "load_checkpoint_dir",

            "start_epoch",
            "n_epochs",

            "log_period",
            "save_period",
        ]

        self.shared_params = [
            "exp_name",
            
            "random_seed",
            
            "dataset_manager",
            
            "model",
        ]

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.data = self.dataset_manager.load_data("train", self.batch_size, self.model.warmup_time, self.train_time_horizon)

        if not self.load_checkpoint_dir:
            assert self.start_epoch == 0, "If not loading from checkpoint, start_epoch should be 0"
        if self.load_checkpoint_dir:
            assert self.start_epoch != 0, "If loading from checkpoint, start_epoch should not be 0"

        self.n_model_params = tf.reduce_sum(tf.stack([tf.size(var) for var in self.model.trainable_variables]))

        self.checkpoint = tf.train.Checkpoint(model=self.model)
        if self.load_checkpoint_dir:
            self.checkpoint.restore(self.load_checkpoint_dir + "ckpts/ckpt-{}".format(self.start_epoch - 1))

        self.summary_writer = tf.summary.create_file_writer(self.exp_name + "/train_log")
        self.metrics = {
            "loss": tf.keras.metrics.Mean(name="loss", dtype=tf.float32),
            "avg_grad_mag": tf.keras.metrics.Mean(name="avg_grad_mag", dtype=tf.float32),
        }

    def train(self):
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)

        with self.summary_writer.as_default():
            for epoch in tqdm.tqdm(range(self.start_epoch, self.start_epoch + self.n_epochs)):
                for data_batch in self.data:
                    self.train_step(data_batch)

                if (epoch + 1) % self.save_period == 0 or (epoch + 1) == (self.start_epoch + self.n_epochs):
                    self.checkpoint.write(self.exp_name + "/ckpts/ckpt-{}".format(epoch))

                if (epoch + 1) % self.log_period == 0 or (epoch + 1) == (self.start_epoch + self.n_epochs):
                    for (metric_name, tf_metric) in self.metrics.items():
                        res = tf_metric.result()
                        print(metric_name, res)
                        tf.summary.scalar(metric_name, res, step=epoch)
                        tf_metric.reset_states()

    @tf.function
    def train_step(self, data_batch):
        with tf.GradientTape() as grad_tape:
            model_pred = self.model.predict(data_batch)
            loss = self.loss.get_loss(data_batch, model_pred)
        gradients = grad_tape.gradient(loss, self.model.trainable_variables)

        self.metrics["loss"].update_state(loss)

        grad_mag_sum = tf.reduce_sum(tf.stack([tf.reduce_sum(tf.abs(gradient)) for gradient in gradients]))
        self.metrics["avg_grad_mag"].update_state(grad_mag_sum / tf.cast(self.n_model_params, tf.float32))

        self.optimizer.tf_optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
