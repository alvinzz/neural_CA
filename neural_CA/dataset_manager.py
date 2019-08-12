import tensorflow as tf
import numpy as np
import glob

from parameter import Parameter

class Dataset_Manager(Parameter):
    param_path = "neural_CA.dataset_manager"
    param_name = "Dataset_Manager"

    def __init__(self):
        self.global_params = set([])
        self.params = set([
            "data_prefix",
            "data_time_horizon",
        ])
        self.shared_params = set([
            "neighbor_rule",
            "obs_dim",
            "random_seed",
        ])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        raise NotImplementedError

    def create_dataset(self):
        raise NotImplementedError

    def parse_example(self):
        raise NotImplementedError

class XOR_Dataset_Manager(Dataset_Manager):
    param_path = "neural_CA.dataset_manager"
    param_name = "XOR_Dataset_Manager"

    def __init__(self):
        self.global_params = set([])
        self.params = set([
            "grid",
            "n_samples",
            "train_val_test_split",
            "data_prefix",
            "shuffle_buffer_size",
            "prefetch_buffer_size",
            "data_time_horizon",
        ])
        self.shared_params = set([
            "neighbor_rule",
            "obs_dim",
            "random_seed",
        ])

        for global_param in self.global_params:
            setattr(self, global_param, None)

        for param in self.params:
            setattr(self, param, None)

    def _build(self):
        self.train_thresh = self.train_val_test_split[0] / sum(self.train_val_test_split)
        self.val_thresh = (self.train_val_test_split[0] + self.train_val_test_split[1]) \
            / sum(self.train_val_test_split)

    def create_dataset(self):
        np.random.seed(self.random_seed)

        train_data = []
        val_data = []
        test_data = []

        for i in range(self.n_samples):
            self.grid.set_random_state()
            grid_obs = []

            for t in range(self.data_time_horizon):
                grid_obs.extend(self.grid.get_obs().flatten().tolist())
                self.grid.update()

            if i / self.n_samples < self.train_thresh:
                train_data.append(grid_obs)
            elif i / self.n_samples < self.val_thresh:
                val_data.append(grid_obs)
            else:
                test_data.append(grid_obs)

        train_examples = map(self.gen_example, train_data)
        val_examples = map(self.gen_example, val_data)
        test_examples = map(self.gen_example, test_data)

        record_file = self.data_prefix + "_train.tfrecords"
        with tf.io.TFRecordWriter(record_file) as writer:
            for example in train_examples:
                writer.write(example.SerializeToString())

        record_file = self.data_prefix + "_val.tfrecords"
        with tf.io.TFRecordWriter(record_file) as writer:
            for example in train_examples:
                writer.write(example.SerializeToString())

        record_file = self.data_prefix + "_test.tfrecords"
        with tf.io.TFRecordWriter(record_file) as writer:
            for example in train_examples:
                writer.write(example.SerializeToString())

    def gen_example(self, grid_obs):
        feature = {
          "grid_obs": tf.train.Feature(float_list=tf.train.FloatList(value=grid_obs)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    def get_parse_example_fn(self, model_warmup_time, pred_time_horizon):
        def parse_example(example):
            feature_description = {
                "grid_obs": tf.io.FixedLenFeature(
                    [self.data_time_horizon * self.neighbor_rule.n_cells * self.obs_dim], tf.float32),
            }
            example = tf.io.parse_single_example(example, feature_description)
           
            example["grid_obs"] = tf.reshape(example["grid_obs"],
                [self.data_time_horizon, self.neighbor_rule.n_cells, self.obs_dim])
           
            example["warmup_obs"] = example["grid_obs"][:model_warmup_time]
            example["pred_gt"] = example["grid_obs"][model_warmup_time:model_warmup_time + pred_time_horizon]

            return example
        
        return parse_example

    def get_add_pred_time_horizon_fn(self, pred_time_horizon):
        def add_pred_time_horizon(example):
            example["pred_time_horizon"] = pred_time_horizon
            return example
        return add_pred_time_horizon

    def load_data(self, split, batch_size, model_warmup_time, pred_time_horizon):
        filenames = glob.glob(self.data_prefix + "_" + split + "*.tfrecords")
        with tf.device("/cpu:0"):
            dataset = tf.data.TFRecordDataset(filenames=filenames)
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.random_seed)
            dataset = dataset.map(self.get_parse_example_fn(model_warmup_time, pred_time_horizon))
            dataset = dataset.batch(batch_size)
            dataset = dataset.map(self.get_add_pred_time_horizon_fn(pred_time_horizon))
            dataset = dataset.prefetch(self.prefetch_buffer_size)
        return dataset
