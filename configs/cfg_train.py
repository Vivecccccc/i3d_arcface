from easydict import EasyDict as edict

config = edict()
config.margin_list = (1.0, 0.0, 0.4)
config.output = './output'
config.dataset_output = './output/train_dataset'
config.data = './data/train_data'
config.pretrain = None
config.max_frame = 64
config.dataset_recorded = False
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 1e-4
config.batch_size = 4
config.lr = 0.1

config.num_classes = 36
config.num_epoch = 20
config.warmup_epoch = 0

config.wandb_key = "x"
config.suffix_run_name = None
config.using_wandb = True
config.wandb_entity = "vivxxxxxxx"
config.wandb_project = "i3d_arcface_pool"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
