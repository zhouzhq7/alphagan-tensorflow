from easydict import EasyDict as edict

config = edict()


"data and check point directories"
config.raw_image_dir = ''
config.data_tfrecord_dir = './train/dataset_middle1.tf'


"optimization"
config.batch_size = 64
config.lr_init = 1e-4
config.beta1 = 0.5
config.beta2 = 0.9

"loss"
config.loss_type = 'sigmoid'
config.recons_loss_w = 40.0

config.n_epoch = 1000
config.lr_decay = 1

"generator type"
config.generator_type = 'dcgan' # or 'dcgan'

config.use_augmentation = True
"summaries"
config.summary_dir = './summary'

config.save_every = 1
config.num_of_update_for_e_g = 2
config.hidden_dim = 128
config.num_of_resblk = 6
