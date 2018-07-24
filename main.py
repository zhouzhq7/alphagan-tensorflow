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

"loss"
loss_type = config.loss_type

"adversarial learning (GAN)"
n_epoch = config.n_epoch
lr_decay = config.lr_decay

"tfrecord data file"
filename = config.data_tfrecord_dir

"summaries"
summary_dir = config.summary_dir

num_of_update_for_e_g = config.num_of_update_for_e_g

recons_loss_w = config.recons_loss_w

save_every_epoch = config.save_every

num_of_resblk = config.num_of_resblk

hidden_dim = config.hidden_dim


def train():

    test_images = get_test_images()

    t_image = tf.placeholder(tf.float32, [None, 64, 64, 3], name='real_image')

    net_e, z_hat = encoder((t_image/127.5)-1, num_of_resblock=num_of_resblk,
                           h_dim=hidden_dim, is_train=True, reuse=False)

    t_z = tf.placeholder(tf.float32, [None, hidden_dim], name='z_prior')

    net_g, x_gen = generator(t_z, is_train=True, reuse=False)

    _, x_recons = generator(z_hat, is_train=True, reuse=True)

    net_cd, cd_logits_fake = code_discriminator(z_hat, reuse=False)

    _, cd_logits_real = code_discriminator(t_z, reuse=True)

    net_d, d_logits_fake1 = discriminator(x_recons, is_train=True, reuse=False)

    _, d_logits_fake2 = discriminator(x_gen, is_train=True, reuse=True)

    _, d_logits_real = discriminator((t_image/127.5)-1, is_train=True, reuse=True)

    "define test network"
    net_e_test, z_test = encoder((t_image/127.5)-1, num_of_resblock=num_of_resblk,
                                 h_dim=hidden_dim, is_train=False, reuse=True)
    net_g_test, _ = generator(z_test, is_train=False, reuse=True)

    "define another test network to evaluate the generative performance of generator"
    net_g_test1, _ = generator(t_z, hidden_dim=hidden_dim, is_train=False, reuse=True)
    np.random.seed(42)
    sampled_z_test = np.random.normal(0.0, 1.0, [16, hidden_dim])

    "auto encoder loss"
    reconstruction_loss = recons_loss_w*tf.reduce_mean(tf.losses.absolute_difference(
        x_recons, (t_image/127.5)-1
    ))

    with tf.name_scope('s_encoder'):
        if loss_type == 'lse':
            e_loss1 = tf.reduce_mean(tf.squared_difference(cd_logits_fake,
                                                           tf.ones_like(cd_logits_fake)))
        else:
            e_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=cd_logits_fake, labels=tf.ones_like(cd_logits_fake)))

        e_loss = e_loss1 + reconstruction_loss

        "define summaries"
        s_e_recons_loss = tf.summary.scalar('reconstruction_loss',
                                            reconstruction_loss)
        s_e_adverse_loss = tf.summary.scalar('adverse_loss', e_loss1)
        s_e_overall_loss = tf.summary.scalar('overall_loss', e_loss)
        e_merge = tf.summary.merge([s_e_recons_loss, s_e_adverse_loss, s_e_overall_loss])
        e_summary_writer = tf.summary.FileWriter(summary_dir+'/encoder')

    with tf.name_scope('s_generator'):
        "generator loss"
        if loss_type == 'lse':
            g_loss1 = tf.reduce_mean(tf.squared_difference(d_logits_fake1,
                                                           tf.ones_like(d_logits_fake1)))
        else:
            g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1,
                                                                     labels=tf.ones_like(d_logits_fake1)))
        if loss_type == 'lse':
            g_loss2 = tf.reduce_mean(tf.squared_difference(d_logits_fake2,
                                                           tf.ones_like(d_logits_fake2)))
        else:
            g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2,
                                                                     labels=tf.ones_like(d_logits_fake2)))

        g_loss = reconstruction_loss + g_loss1 + g_loss2

        "define summaries"
        s_g_adverse_recons_loss = tf.summary.scalar('adverse_recons_loss', g_loss1)
        s_g_adverse_gen_loss = tf.summary.scalar('adverse_gen_loss', g_loss2)
        s_g_reconstruction_loss = tf.summary.scalar('reconstruction_loss',
                                                    reconstruction_loss)
        s_g_overall_loss = tf.summary.scalar('overall_loss', g_loss)

        g_merge = tf.summary.merge([s_g_adverse_gen_loss, s_g_adverse_recons_loss, s_g_reconstruction_loss, s_g_overall_loss])
        g_summary_writer = tf.summary.FileWriter(summary_dir+'/generator')


    with tf.name_scope('s_discriminator'):
        "discriminator loss"
        if loss_type == 'lse':
            d_loss1 = tf.reduce_mean(tf.squared_difference(d_logits_fake1,
                                                           tf.zeros_like(d_logits_fake1)))
        else:
            d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake1,
                                                              labels=tf.zeros_like(d_logits_fake1)))

        if loss_type == 'lse':
            d_loss2 = tf.reduce_mean(tf.squared_difference(d_logits_fake2,
                                                           tf.zeros_like(d_logits_fake2)))
        else:
            d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake2,
                                                              labels=tf.zeros_like(d_logits_fake2)))

        if loss_type == 'lse':
            d_loss3 = tf.reduce_mean(tf.squared_difference(d_logits_real,
                                                           tf.ones_like(d_logits_real)))
        else:
            d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                              labels=tf.ones_like(d_logits_real)))

        d_loss = d_loss1 + d_loss2 + d_loss3

        "define summaries"
        s_d_adverse_recons_loss = tf.summary.scalar('adverse_recons_loss', d_loss1)
        s_d_adverse_gen_loss = tf.summary.scalar('adverse_gen_loss', d_loss2)
        s_d_real_loss = tf.summary.scalar('adverse_real_loss', d_loss3)
        s_d_overall_loss = tf.summary.scalar('overall_loss', d_loss)

        d_merge = tf.summary.merge([s_d_adverse_gen_loss, s_d_adverse_recons_loss, s_d_real_loss, s_d_overall_loss])
        d_summary_writer = tf.summary.FileWriter(summary_dir+'/discriminator')

    with tf.name_scope("s_code_discriminator"):
        "code discriminator loss"
        if loss_type == 'lse':
            cd_loss1 = tf.reduce_mean(tf.squared_difference(cd_logits_fake,
                                                            tf.zeros_like(cd_logits_fake)))
        else:
            cd_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_fake,
                                                               labels=tf.zeros_like(cd_logits_fake)))

        if loss_type == 'lse':
            cd_loss2 = tf.reduce_mean(tf.squared_difference(cd_logits_real,
                                                            tf.ones_like(cd_logits_real)))
        else:
            cd_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=cd_logits_real,
                                                               labels=tf.ones_like(cd_logits_real)))
        cd_loss = cd_loss1 + cd_loss2

        s_cd_adverse_loss_fake = tf.summary.scalar('adverse_loss_fake', cd_loss1)
        s_cd_adverse_loss_real = tf.summary.scalar('adverse_loss_real', cd_loss2)
        s_cd_overall_loss = tf.summary.scalar('overall_loss', cd_loss)

        cd_merge = tf.summary.merge([s_cd_adverse_loss_fake, s_cd_adverse_loss_real, s_cd_overall_loss])
        cd_summary_writer = tf.summary.FileWriter(summary_dir+'/code_discriminator')

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


    save_gan_dir = "./samples/train_gan"
    save_test_gan_dir = "./samples/test_gan"
    checkpoints_dir = "./checkpoints"

    mkdir_if_not_exists(save_gan_dir)
    mkdir_if_not_exists(checkpoints_dir)
    mkdir_if_not_exists(save_test_gan_dir)

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


    num_of_data = 20000
    num_of_iter_one_epoch = num_of_data // batch_size

    sess.run(tf.assign(lr_v, lr_init))
    print ("Traing alpha-GAN with initialized learning rate: %f" % (lr_init))

    img_batch = inputs(filename, batch_size, n_epoch, shuffle_size=20000,
                       is_augment=False, is_resize=True)
    try:
        epoch_time = time.time()
        n_iter = 0
        while True:
            if (n_iter + 1) % (num_of_iter_one_epoch) == 0:
                log = "[*] Epoch [%4d/%4d] time: %4.4fs" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch, time.time()-epoch_time
                )
                print (log)
                lr_new = lr_init * (lr_decay**((n_iter+1)//num_of_iter_one_epoch))
                print ((lr_decay**((n_iter+1)//num_of_iter_one_epoch)))
                sess.run(tf.assign(lr_v, lr_new))
                print ("Traing alpha-GAN with new learning rate: %f" % (lr_new))
                epoch_time = time.time()

            step_time = time.time()

            imgs = np.array(sess.run(img_batch))

            batch_sz = imgs.shape[0]
            "sample a standard normal distribution"
            z_prior = np.random.normal(0, 1.0, (batch_sz, hidden_dim))

            "update encoder and generator multiple times"
            for i in range(num_of_update_for_e_g):
                "update encoder"
                e_summary, err_E_recons_loss, err_E_adversarial_loss, err_E_loss, _ = sess.run(
                    [e_merge, reconstruction_loss, e_loss1, e_loss, e_optim], feed_dict={t_image: imgs, t_z: z_prior})

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, e_loss: %8f, e_recons_loss: %8f, e_adverse_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_E_loss, err_E_recons_loss,
                    err_E_adversarial_loss
                )

                print (log)

                e_summary_writer.add_summary(e_summary, n_iter*num_of_iter_one_epoch + i)

                "update generator"
                g_summary, err_G_recons_loss, err_G_adverse_loss, err_G_gen_loss, err_G_loss, _ = sess.run(
                    [g_merge, reconstruction_loss, g_loss1, g_loss2, g_loss, g_optim], feed_dict={t_image:imgs, t_z: z_prior}
                )

                log = "Epoch [%4d/%4d] %6d time: %4.4fs, g_loss: %8f, g_recons_loss: %8f, g_adverse_loss: %8f, g_gen_loss: %8f" % (
                    (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_G_loss, err_G_recons_loss,
                    err_G_adverse_loss, err_G_gen_loss
                )

                print (log)

                g_summary_writer.add_summary(g_summary, n_iter*num_of_iter_one_epoch + i)

            "update discriminator"
            d_summary, err_D_real_loss, err_D_recons_loss, err_D_gen_loss, err_D_loss, _ = \
                sess.run([d_merge, d_loss3, d_loss1, d_loss2, d_loss, d_optim],
                                                                             feed_dict={t_image:imgs, t_z: z_prior})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, d_loss: %8f, d_recons_loss: %8f, d_gen_loss: %8f, d_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_D_loss, err_D_recons_loss,
                err_D_gen_loss, err_D_real_loss
            )
            print (log)

            d_summary_writer.add_summary(d_summary, n_iter)

            "update code discriminator"

            cd_summary, err_CD_fake_loss, err_CD_real_loss, err_CD_loss, _ = \
                sess.run([cd_merge, cd_loss1, cd_loss2, cd_loss, cd_optim],
                                                                                         feed_dict={t_image:imgs, t_z: z_prior})

            log = "Epoch [%4d/%4d] %6d time: %4.4fs, cd_loss: %8f, cd_fake_loss: %8f, cd_real_loss: %8f" % (
                (n_iter+1)//num_of_iter_one_epoch, n_epoch,n_iter, time.time() - step_time, err_CD_loss, err_CD_fake_loss,
                err_CD_real_loss
            )

            print (log)

            cd_summary_writer.add_summary(cd_summary, n_iter)

            if ((n_iter+1) % (save_every_epoch * num_of_iter_one_epoch) == 0):
                tl.files.save_npz(net_g.all_params,
                                  name=checkpoints_dir + '/g_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_e.all_params,
                                  name=checkpoints_dir + '/e_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_d.all_params,
                                  name=checkpoints_dir + '/d_{}.npz'.format(tl.global_flag['mode']), sess=sess)
                tl.files.save_npz(net_cd.all_params,
                                  name=checkpoints_dir + '/cd_{}.npz'.format(tl.global_flag['mode']), sess=sess)

            if ( (n_iter + 1) % (num_of_iter_one_epoch * save_every_epoch) == 0):
                # quick evaluation on train set
                out = sess.run(net_g_test.outputs,
                               {t_image: test_images})
                out = (out+1)*127.5
                print ("reconstructed image:", out.shape, out.min(), out.max())
                print("[*] save images")
                tl.vis.save_images(out.astype(np.uint8), [4, 4], save_gan_dir +
                                   '/train_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))

                # quick evaluation on generative performance of generator
                out1 = sess.run(net_g_test1.outputs, feed_dict={t_z: sampled_z_test})
                out1 = (out1+1)*127.5
                print ("generated image:", out1.shape, out1.min(), out1.max())
                print("[*] save images")
                tl.vis.save_images(out1.astype(np.uint8), [4, 4], save_test_gan_dir
                                   + '/test_%d.png' % ((n_iter + 1) // num_of_iter_one_epoch))

            n_iter += 1

    except tf.errors.OutOfRangeError:
        print ("training is done")
        pass

def evaluate(sub='generator', num=16):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    result_dir = './results'
    checkpoints_dir = "./checkpoints"
    mkdir_if_not_exists(result_dir)
    if sub == 'generator':
        with tf.name_scope('evaluation'):
            t_z = tf.placeholder(tf.float32, [num, hidden_dim], name='test_sampled_prior')
            net_g_test = generator(t_z, is_train=False, reuse=False)

            tl.layers.initialize_global_variables(sess)

            tl.files.load_and_assign_npz(sess=sess,
                                         name=checkpoints_dir+"/g_{}.npz".format(tl.global_flag['mode']),
                                         network=net_g_test)
            sampled_z = np.random.normal(0.0, 1.0, [num, hidden_dim])
            out = sess.run(net_g_test.outputs,
                           {t_z: sampled_z})
            out = (out+1)*127.5
            print ("gen sub image:", out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out.astype(np.uint8), [4, 4], result_dir+ '/test_'+sub+'.png')

if __name__== "__main__":

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if args.mode == 'train':
        train()
    elif args.mode == 'evaluate':
        evaluate()
    else:
        raise Exception('Unknow mode {}'.format(args.mode))
