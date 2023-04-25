from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.data = './data/test_data'
config.output = './output'
config.dataset_output = './output/test_dataset'
config.pretrain = './output/model_final.pt'
config.dataset_recorded = False
config.embedding_size = 512
config.batch_size = 4
config.num_classes = 36
config.return_embedding = True
