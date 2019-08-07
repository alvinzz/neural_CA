# ml_pipeline

Pipeline for ML training that records (hyper-)parameters and allows restoring a model using those (hyper-)parameters.

Example usage (requires TensorFlow 2.0)*:
```
export PYTHONPATH=$PYTHONPATH:{PATH_TO_ml_pipeline}
cd xor_example
```

In this example, we will be training a MLP on a toy XOR dataset.

To run our first experiment, follow these steps:

1. Make sure that lines 64-77 of `run_xor_experiment.py` appear as follows:
```
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
```

2. Run `python run_xor_experiment.py`. 

Let's dive into what is happening here.

1. First, the `setup_experiment()` function creates an `Experiment` object with `Model`, `Trainer`, and `Evaluator` attributes, which have further sub-attributes and hyper-parameters. See `run_xor_experiment.py` for further details. 
```
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
            ...

        XOR_evaluator = Evaluator()

        XOR_experiment.model = XOR_model
        XOR_experiment.trainer = XOR_trainer
        XOR_experiment.evaluator = XOR_evaluator

    return XOR_experiment
```

2. Then, the `Experiment.set_exp_name("xor")` method stores the `"xor"` string for logging purposes.

3. Next, the `Experiment.train()` method calls `Trainer.train(Model)`, which trains the model using the hyper-parameters of the `Trainer`.

4. Lastly, the `Experiment.save()` method then saves all of the hyper-parameters of the `Experiment` into a `params` file. This is stored under the `xor_YYYY_MM_DD_HH_MM_SS` folder, where the `"xor"` prefix was set with the `Experiment.set_exp_name` method.

After the experiment has finished running, your working directory should look like this:
```
xor_example
+-- run_xor_experiment.py
+-- xor_dataset_utils.py
+-- last_params
+-- xor_YYYY_MM_DD_HH_MM_SS
|   +-- params
|   +-- train_log
|       +-- events.out.tfevents.*
|   +-- ckpts
|       +-- ckpt-49.index
|       +-- ckpt-49.data-00000-of-00001
```

The `last_params` file is a copy of the `xor_YYYY_MM_DD_HH_MM_SS/params` file, and is created for convenience.

Running `tensorboard --logdir xor_YYYY_MM_DD_HH_MM_SS/train_log/` shows that training has been slow. 

![exp1_loss](/doc_images/exp1_loss.png)

In order to pick up training where we left off, with different hyper-parameters, follow these steps:

1. First, edit `last_params`. Change `trainer/load_checkpoint_dir` to the `"xor_YYYY_MM_DD_HH_MM_SS/"` folder, `trainer/start_epoch` to `50`, and `trainer/optimizer/epsilon` to `1e-7`.
```
last_params
{
  "param_path": "experiment",
  "param_name": "Experiment",
  "model": {
    ...
  },
  "trainer": {
    ...
    "optimizer": {
      "param_path": "trainers.tf_utils.optimizers",
      "param_name": "TF_Adam_Optimizer",
      "learning_rate": 0.01,
      "epsilon": 0.1 => 1e-7
    },
    "load_checkpoint_dir": null => "xor_YYYY_MM_DD_HH_MM_SS/",
    "start_epoch": 0 => 50,
    "n_epochs": 50,
    "batch_size": 4,
    "log_period": 1,
    "save_period": 50
  },
  "evaluator": {
    ...
  }
}
```

2. Next, edit `run_xor_experiment.py`. Change lines 64-77 to:
```
if __name__ == "__main__":
    create_xor_dataset()

    # setup experiment for the first time
    #XOR_experiment = setup_experiment()
    # restore experiment from a params file
    XOR_experiment = Experiment()
    XOR_experiment.load("last_params")

    XOR_experiment.set_exp_name("xor")

    XOR_experiment.train()

    XOR_experiment.save()
```

3. Now, running `python run_xor_experiment.py` will run an experiment with the new hyper-parameters in `last_params`. It also creates a new folder, which contains the `params`, logs, and checkpoints of the new experiment. 
```
xor_example
+-- run_xor_experiment.py
+-- xor_dataset_utils.py
+-- last_params
+-- xor_YYYY_MM_DD_HH_MM_SS (new)
|   +-- params
|   +-- train_log
|       +-- events.out.tfevents.*
|   +-- ckpts
|       +-- ckpt-99.index
|       +-- ckpt-99.data-00000-of-00001
+-- xor_YYYY_MM_DD_HH_MM_SS (old)
|   +-- params
|   +-- train_log
|       +-- events.out.tfevents.*
|   +-- ckpts
|       +-- ckpt-49.index
|       +-- ckpt-49.data-00000-of-00001
```

Running TensorBoard (`tensorboard --logdir xor_YYYY_MM_DD_HH_MM_SS (new)/train_log/`) shows that the model has now converged with the new hyper-parameters.

![exp2_loss](/doc_images/exp2_loss.png)

*Install with `pip install tensorflow==2.0.0-beta1` or `pip install tensorflow-gpu==2.0.0-beta1`.
