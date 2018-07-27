from easydict import EasyDict as edict

config = edict()


"data and check point directories"
config.raw_image_dir = ''
config.data_tfrecord_dir = ['./train/fornite_train.tf', './train/fornite_test.tf', './train/pubg_train.tf', './train/pubg_test.tf']


"optimization"
config.batch_size = 32
config.lr_init = 2e-4
config.beta1 = 0.5
config.beta2 = 0.9

"loss"
config.loss_type = 'sigmoid'
config.recons_loss_w = 40.0
config.e_adverse_loss_w = 8.0
config.g_gen_loss_w = 8.0

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
config.num_of_resblk = 9

config.num_of_data = 5000
