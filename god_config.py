from easydict import EasyDict as edict

config = edict()


"data and check point directories"
config.raw_image_dir = ''
config.data_tfrecord_dir = './train/dataset_middle.tf'


"optimization"
config.batch_size = 32
config.lr_init = 1e-4
config.beta1 = 0.9

"loss"
config.loss_type = 'lse'
config.recons_loss_w = 20.0
"g initialization"
config.n_epoch_init = 0

config.n_epoch = 1000
config.lr_decay = 0.995

"generator type"
config.generator_type = 'resblk' # or 'dcgan'

"summaries"
config.summary_dir = './summary'

config.save_every = 5
config.num_of_update_for_e_g = 2
config.hidden_dim = 128
config.num_of_resblk = 6
