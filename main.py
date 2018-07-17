from models import *
from god_config import *
from utils import *
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--image_raw_data_dir', type=str, default=config.raw_image_dir,
                    help='directory contains all raw images with size (224, 224, 3)')

parser.add_argument('--mode', type=str, default='train',
                    help='train or evaluate')

"optimizer"
batch_size = config.batch_size
lr_init = config.lr_init
beta1 = config.beta1

"initialize g"
n_epoch_init = config.n_epoch_init

"adversarial learning (GAN)"
n_epoch = config.n_epoch
lr_decay = config.lr_decay
decay_every = config.decay_every

"tfrecord data file"
filename = config.data_tfrecord_dir

save_every_epoch = 1

def train():

    test_images = get_test_images()

    hidden_dim = 512

    t_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_image')

    net_e, z_hat = encoder(t_image, is_train=True, reuse=False)

    t_z = tf.placeholder(tf.float32, [None, hidden_dim], name='z_prior')

    net_g, x_gen = generator(t_z, is_train=True, reuse=False)

    _, x_recons = generator(z_hat, is_train=True, reuse=True)

    net_cd, cd_logits_fake = code_discriminator(z_hat, reuse=False)

    _, cd_logits_real = code_discriminator(t_z, reuse=True)

    net_d, d_logits_fake1 = discriminator(x_recons, is_train=True, reuse=False)

    _, d_logits_fake2 = discriminator(x_gen, is_train=True, reuse=True)

    _, d_logits_real = discriminator(t_image, is_train=True, reuse=True)

    "define test network"
    net_e_test, z_test = encoder(t_image, is_train=False, reuse=True)
    net_g_test, _ = generator(z_test, is_train=False, reuse=True)

    "auto encoder loss"
    reconstruction_loss = tf.reduce_mean(tf.losses.absolute_difference(
        x_recons, t_image
    ))

    e_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_fake,
                                                                      labels=tf.ones_like(cd_logits_fake)))
    e_loss = e_loss1 + reconstruction_loss

    "generator loss"

    g_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1,
                                                             labels=tf.ones_like(d_logits_fake1))

    g_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2,
                                                             labels=tf.ones_like(d_logits_fake2))

    g_loss = reconstruction_loss + g_loss1 + g_loss2

    "discriminator loss"

    d_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1,
                                                      labels=tf.zeros_like(d_logits_fake1))

    d_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2,
                                                      labels=tf.zeros_like(d_logits_fake2))

    d_loss3 = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                      labels=tf.ones_like(d_logits_real))

    d_loss = d_loss1 + d_loss2 + d_loss3

    "code discriminator loss"

    cd_loss1 = tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_fake,
                                                       labels=tf.zeros_like(cd_logits_fake))

    cd_loss2 = tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_real,
                                                       labels=tf.ones_like(cd_logits_real))

    cd_loss = cd_loss1 + cd_loss2

    e_vars = tl.layers.get_variables_with_name(name='encoder', train_only=True, printable=True)
    g_vars = tl.layers.get_variables_with_name(name='generator', train_only=True, printable=True)
    d_vars = tl.layers.get_variables_with_name(name='discriminator', train_only=True, printable=True)
    cd_vars = tl.layers.get_variables_with_name(name='code_evaluator', train_only=True, printable=True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(e_loss, var_list=e_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    cd_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(cd_loss, var_list=cd_vars)


    save_gan_dir = "./samples/{}_gan".format(tl.global_flag['mode'])
    checkpoints_dir = "./checkpoints"
    pre_trained_model_dir = "./models"

    mkdir_if_not_exists(save_gan_dir)
    mkdir_if_not_exists(checkpoints_dir)
    mkdir_if_not_exists(pre_trained_model_dir)


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))

    tl.layers.initialize_global_variables(sess)

    tl.files.load_and_assign_npz(sess=sess,
                                    name=checkpoints_dir+"/g_{}.npz".format(tl.global_flag['mode']),
                                    network=net_g)

    tl.files.load_and_assign_npz(sess=sess,
                                     name=checkpoints_dir+"/e_{}.npz".format(tl.global_flag['mode']),
                                     network=net_e)

    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoints_dir+"/cd_{}.npz".format(tl.global_flag['mode']),
                                 network=net_cd)

    tl.files.load_and_assign_npz(sess=sess,
                                 name=checkpoints_dir+"/d_{}.npz".format(tl.global_flag['mode']),
                                 network=net_d)

    img_batch = inputs(filename, batch_size, n_epoch, shuffle_size=500, is_augment=False, is_resize=True)


    num_of_data = 4000
    num_of_iter_one_epoch = num_of_data // batch_size

    sess.run(tf.assign(lr_v, lr_init))
    print ("Traing alpha-GAN with initialized learning rate: %f" % (lr_init))

    try:
        epoch_time = time.time()
        n_iter = 0
        while True:
            if (n_iter + 1) % (num_of_iter_one_epoch) == 0:
                log = "[*] Epoch [%4d/%4d] time: %4.4fs" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, time.time()-epoch_time
                )
                print (log)
                epoch_time = time.time()

            step_time = time.time()

            imgs = np.array(sess.run(img_batch))

            batch_sz = imgs.shape[0]
            "sample a standard normal distribution"
            z_prior = np.random.normal(0, 1.0, (batch_sz, hidden_dim))
            "update encoder and generator multiple times"
            for i in range(1):
                "update encoder"
                err_E_recons_loss, err_E_adversarial_loss, err_E_loss, _ = sess.run(
                    [reconstruction_loss, e_loss1, e_loss, e_optim], feed_dict={t_image: imgs, t_z: z_prior})

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, e_loss: %8f, e_recons_loss: %8f, e_adverse_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_E_loss, err_E_recons_loss,
                    err_E_adversarial_loss
                )

                print (log)

                "update generator"
                err_G_recons_loss, err_G_adverse_loss, err_G_gen_loss, err_G_loss, _ = sess.run(
                    [reconstruction_loss, g_loss1, g_loss2, g_loss, g_optim], feed_dict={t_image:imgs, t_z: z_prior}
                )

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, g_loss: %8f, g_recons_loss: %8f, g_adverse_loss: %8f, g_gen_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_G_loss, err_G_recons_loss,
                    err_G_adverse_loss, err_G_gen_loss
                )

                print (log)

            "update discriminator"
            err_D_real_loss, err_D_recons_loss, err_D_gen_loss, err_D_loss, _ = sess.run([d_loss3, d_loss1, d_loss2, d_loss, d_optim],
                                                                             feed_dict={t_image:imgs, t_z: z_prior})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, d_loss: %8f, d_recons_loss: %8f, d_gen_loss: %8f, d_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_D_loss, err_D_recons_loss,
                err_D_gen_loss, err_D_real_loss
            )

            print (log)

            "update code discriminator"

            err_CD_fake_loss, err_CD_real_loss, err_CD_loss, _ = sess.run([cd_loss1, cd_loss2, cd_loss, cd_optim],
                                                                                         feed_dict={t_image:imgs, t_z: z_prior})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, cd_loss: %8f, cd_fake_loss: %8f, cd_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_CD_loss, err_CD_fake_loss,
                err_CD_real_loss
            )

            print (log)

            # quick evaluation on train set
            if ( (n_iter + 1) % (num_of_iter_one_epoch * save_every_epoch) == 0):
                out = sess.run(net_g_test.outputs,
                               {t_image: test_images})
                out = (out+1)*127.5
                print ("gen sub image:", out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out.astype(np.uint8), [4, 4], save_gan_dir + '/train_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))


    except tf.errors.OutOfRangeError:
        print ("training is done")
        pass

def evaluate():
    pass

if __name__== "__main__":

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        raise Exception('Unknow mode {}'.format(args.mode))
