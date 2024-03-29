from neural_CA.ca_experiment import CA_Experiment

ca_exp = CA_Experiment()
ca_exp.obs_dim = 1
# if True:
#     from neighbor_rule import Rect_Neighbor_Rule
#     neighbor_rule = Rect_Neighbor_Rule()
#     neighbor_rule.grid_width = 5
#     neighbor_rule.grid_height = 5
#     neighbor_rule.grid_connectivity = 8
if True:
    from neighbor_rule import Hex_Neighbor_Rule
    neighbor_rule = Hex_Neighbor_Rule()
    neighbor_rule.grid_radius = 5
ca_exp.neighbor_rule = neighbor_rule
ca_exp.exp_name = "test"
ca_exp.random_seed = 0
if True:
    from models.tf_grid_model import TF_Grid_Model_v1
    model = TF_Grid_Model_v1()
    model.model_hidden_dim = 2
    model.effect_dotp_dim = 6
    model.effect_dotp_MLP_hidden_sizes = []
    model.effect_dim = 12
    model.effect_MLP_hidden_sizes = []
    model.apply_dotp_dim = 6
    model.apply_dotp_MLP_hidden_sizes = []
    model.apply_MLP_hidden_sizes = [100 for _ in range(5)]
    model.activation = "relu"
    model.warmup_time = 3
ca_exp.model = model
if True:
    from trainers.tf_grid_trainer import TF_Grid_Trainer
    trainer = TF_Grid_Trainer()
    if True:
        from trainers.tf_utils.losses import TF_Grid_L1_Loss
        loss = TF_Grid_L1_Loss()
        loss.discount = 0.5
    trainer.loss = loss
    if True:
        from trainers.tf_utils.optimizers import TF_Adam_Optimizer
        optimizer = TF_Adam_Optimizer()
        optimizer.learning_rate = 0.00001
        optimizer.epsilon = 1e-7
    trainer.optimizer = optimizer
    trainer.load_checkpoint_dir = "test_2019_08_23_18_29_45/" #"test_2019_08_23_16_46_42/"
    trainer.start_epoch = 1650
    trainer.n_epochs = 1000
    trainer.batch_size = 1000
    trainer.log_period = 1
    trainer.save_period = 50
    trainer.pred_time_horizon = 7
ca_exp.trainer = trainer
ca_exp.evaluator = None
if True:
    from dataset_manager import XOR_Dataset_Manager
    dataset_manager = XOR_Dataset_Manager()
    dataset_manager.data_prefix = "data/xor"
    if True:
        from grid import Grid
        grid = Grid()
        grid.grid_hidden_dim = 2
        if True:
            from cell_ruleset import XOR_Cell_Ruleset
            cell_ruleset = XOR_Cell_Ruleset()
        grid.cell_ruleset = cell_ruleset
    dataset_manager.grid = grid
    dataset_manager.n_samples = 1000
    dataset_manager.train_val_test_split = [0.8, 0.1, 0.1]
    dataset_manager.shuffle_buffer_size = 100
    dataset_manager.prefetch_buffer_size = 100
    dataset_manager.data_time_horizon = 10
ca_exp.dataset_manager = dataset_manager
ca_exp.grid_visualizer = None
ca_exp.param_build()

ca_exp.dataset_manager.create_dataset()

ca_exp.save()

ca_exp.trainer.train()
