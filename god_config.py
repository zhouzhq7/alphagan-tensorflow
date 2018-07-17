from easydict import EasyDict as edict

config = edict()


"data and check point directories"
config.raw_image_dir = ''
config.data_tfrecord_dir = './train/images150.tf'


"optimization"
config.batch_size = 128
config.lr_init = 5e-6
config.beta1 = 0.9

"g initialization"
config.n_epoch_init = 50

config.n_epoch = 100
config.lr_decay = 0.1
config.decay_every = int(config.n_epoch/2)
