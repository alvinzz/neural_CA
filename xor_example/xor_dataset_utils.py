import tensorflow as tf
import numpy as np

from trainers.tf_utils.tf_example_parser import TF_Example_Parser

class XOR_Example_Parser(TF_Example_Parser):
    param_path = "xor_example.xor_dataset_utils"
    param_name = "XOR_Example_Parser"

    def update_parameters(self):
        self.params = {
            "param_path": XOR_Example_Parser.param_path,
            "param_name": XOR_Example_Parser.param_name,
        }

    def parse_example(self, example):
        feature_description = {
            "feature": tf.io.FixedLenFeature([2], tf.float32),
            "label": tf.io.FixedLenFeature([1], tf.float32),
        }
        return tf.io.parse_single_example(example, feature_description)

def create_xor_dataset():
    x = np.array([[0,0], [0,1], [1,0], [1,1]]).astype(np.float32)
    y = np.array([[0], [1], [1], [0]]).astype(np.float32)

    def gen_example(x, y):
        feature = {
          "feature": tf.train.Feature(float_list=tf.train.FloatList(value=x.tolist())),
          "label": tf.train.Feature(float_list=tf.train.FloatList(value=y.tolist())),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))

    record_file = "data/xor.tfrecords"
    with tf.io.TFRecordWriter(record_file) as writer:
        for (_x, _y) in zip(x, y):
            example = gen_example(_x, _y)
            writer.write(example.SerializeToString())
