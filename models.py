import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def encoder(rgb, is_train=True, reuse=False):
    h_dim = 512

    num_of_resblock = 8

    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
    b_init = None
    g_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)


    filter_size = (3, 3)
    strides = (1, 1)

    with tf.variable_scope('encoder', reuse=reuse):

        tl.layers.set_name_reuse(reuse)

        net = InputLayer(rgb, name='e/in')

        net = Conv2d(net, n_filter=64, filter_size=filter_size, strides=strides, act=tf.nn.relu,
                     padding='SAME', W_init=w_init, name='e/n64s1/c0')

        temp = net

        ################################# Residual block ############################################
        for i in range(num_of_resblock):
            net_r = Conv2d(net, n_filter=64, filter_size=filter_size, strides=strides, act=tf.identity,
                           padding='SAME', W_init=w_init, b_init=b_init, name='e/n64s1/c1/%s' % i)

            net_r = BatchNormLayer(net_r, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='e/n64s1/b1/%s' % i)

            net_r = Conv2d(net_r, n_filter=64, filter_size=filter_size, strides=strides, act=tf.identity,
                           padding='SAME', W_init=w_init, b_init=b_init, name='e/n64s1/c2/%s' % i)

            net_r = BatchNormLayer(net_r, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='e/n64s1/b2/%s' % i)

            net_r = ElementwiseLayer([net, net_r], combine_fn=tf.add, name='e/residual_add/%s' % i)

            net = net_r

        net = Conv2d(net, n_filter=64, filter_size=filter_size, strides=strides, act=tf.identity,
                     W_init=w_init, b_init=b_init, name='e/n64s1/c/m')

        net = BatchNormLayer(net, is_train=is_train, gamma_init=g_init, name='e/n64s1/b/m')

        net = ElementwiseLayer([net, temp], combine_fn=tf.add, name='e/add3')

        ################################### End of residual block ####################################
        net = FlattenLayer(net, name='e/flatten')
        net = DenseLayer(net, n_units=h_dim, act=tf.identity, W_init=w_init, name='e/lin/output')

        logits = net.outputs

    return net, logits

def generator(feat_vec, is_train=True, reuse=False):

    w_init = tf.truncated_normal_initializer(stddev=0.02)
    g_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    image_size = 64

    s2, s4, s8, s16 = int(image_size/2), int(image_size/4), int(image_size/8), int(image_size/16),

    gf_dim = 64

    c_dim = 3

    assert feat_vec.get_shape().as_list()[1:] == [512]

    # make sure the size matches if the size of current batch is not batch size
    batch_size = feat_vec.get_shape().as_list()[0]

    filter_size = (5, 5)
    strides = (2, 2)

    with tf.variable_scope('generator', reuse=reuse):

        tl.layers.set_name_reuse(reuse)

        # (512,)
        net_in = InputLayer(feat_vec, name='g/in')

        # (64*16*4*4=4*4*1024, )
        net_h0 = DenseLayer(net_in, n_units=gf_dim*16*s16*s16, W_init=w_init,
                            act = tf.identity, name='g/h0/lin')


        # (4,4, 1024)
        net_h0 = ReshapeLayer(net_h0, shape=[-1, s16, s16, gf_dim*16], name="g/h0/reshape")

        net_h0 = BatchNormLayer(net_h0, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h0/batch_norm')

        # (8, 8, 512)
        net_h1 = DeConv2d(net_h0, n_filter=gf_dim*8, filter_size=filter_size, out_size=(s8, s8), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h1/deconv2d')

        net_h1 = BatchNormLayer(net_h1, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h1/batch_norm')

        # (16, 16, 256)
        net_h2 = DeConv2d(net_h1, n_filter=gf_dim*4, filter_size=filter_size, out_size=(s4, s4), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h2/deconv2d')

        net_h2 = BatchNormLayer(net_h2, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h2/batch_norm')

        # (32, 32, 128)
        net_h3 = DeConv2d(net_h2, n_filter=gf_dim*2, filter_size=filter_size, out_size=(s2, s2), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h3/deconv2d')

        net_h3 = BatchNormLayer(net_h3, act=tf.nn.relu, is_train=is_train,
                                gamma_init=g_init, name='g/h3/batch_norm')

        # (64, 64, 3)
        net_h4 = DeConv2d(net_h3, n_filter=c_dim, filter_size=filter_size, out_size=(image_size, image_size), strides=strides,
                          padding='SAME', batch_size=batch_size, W_init=w_init, name='g/h4/deconv2d')

        net_h4.outputs = tf.nn.tanh(net_h4.outputs)

        return net_h4, net_h4.outputs

def discriminator(inputs, is_train=True, reuse=False):

    df_dim = 64

    w_init = tf.truncated_normal_initializer(stddev=0.02)

    gamma_init = tf.truncated_normal_initializer(mean=1.0, stddev=0.02)

    filter_size = (5, 5)

    strides = (2, 2)

    with tf.variable_scope('discriminator', reuse=reuse):

        tl.layers.set_name_reuse(reuse)

        net_in = InputLayer(inputs=inputs, name='d/in')

        # (32, 32, 64)
        net_h0 = Conv2d(net_in, n_filter=df_dim, filter_size=filter_size, strides=strides,
                        act = lambda x: tl.act.lrelu(x, 0.2), padding='SAME', W_init=w_init, name='d/h0/conv2d')


        # (16, 16, 128)
        net_h1 = Conv2d(net_h0, n_filter=df_dim*2, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h1/conv2d')

        net_h1 = BatchNormLayer(net_h1, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h1/batch_norm')

        # (8, 8, 256)
        net_h2 = Conv2d(net_h1, n_filter=df_dim*4, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h2/conv2')

        net_h2 = BatchNormLayer(net_h2, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h2/batch_norm')

        # (4, 4, 512)
        net_h3 = Conv2d(net_h2, n_filter=df_dim*8, filter_size=filter_size, strides=strides,
                        act=None, padding='SAME', W_init=w_init, name='d/h3/conv2')

        net_h3 = BatchNormLayer(net_h3, act=lambda x:tl.act.lrelu(x, 0.2), is_train=is_train,
                                gamma_init=gamma_init, name='d/h3/batch_norm')

        net_h4 = FlattenLayer(net_h3, name='d/h4/flatten')
        net_h4 = DenseLayer(net_h4, n_units=1, act=tf.identity, W_init=w_init, name='d/h4/lin_sigmoid')

        logits = net_h4.outputs

    return net_h4, logits

def code_discriminator(code, reuse=False):

    h = 4096

    w_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

    with tf.variable_scope('code_evaluator', reuse=reuse):

        tl.layers.set_name_reuse(reuse)

        net = InputLayer(code, name='cd/in')

        net = DenseLayer(net, n_units=h, act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init,
                         name="cd/fc1")

        net = DenseLayer(net, n_units=h, act=lambda x: tl.act.lrelu(x, 0.2), W_init=w_init,
                         name="cd/fc2")

        net = DenseLayer(net, n_units=1, act=tf.identity, W_init=w_init,
                         name="cd/sigmoid")

        logits = net.outputs

    return net, logits






