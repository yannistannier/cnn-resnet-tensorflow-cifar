import tensorflow as tf


def conv2d(scope, input_layer, output_dim, use_bias=False, filter_size=3, strides=(1, 1)):

    conv = tf.layers.conv2d(
        input_layer,
        output_dim,
        kernel_size=filter_size,
        strides=strides,
        padding='same',
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
        kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002)
    )

    return conv




def batch_norm(scope, input_layer, is_training, reuse):
    output_layer = tf.contrib.layers.batch_norm(
        input_layer,
        decay=0.9,
        scale=True,
        epsilon=1e-5,
        is_training=is_training,
        reuse=reuse,
        scope=scope
    )

    return output_layer


def lrelu(input_layer, leak=0.2):
    output_layer = tf.nn.relu(input_layer)
    return output_layer


def fully_connected(scope, input_layer, output_dim):
    fc = tf.layers.dense(input_layer,
                         output_dim,
                         activation=None,
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002),
                         kernel_initializer=tf.contrib.layers.variance_scaling_initializer())
    return fc



def avg_pool(scope, input_layer, ksize=None, strides=[1, 2, 2, 1]):
    if ksize is None:
        ksize = strides

    with tf.variable_scope(scope):
        output_layer = tf.nn.avg_pool(input_layer, ksize, strides, 'VALID')
        return output_layer





def residual(scope, input_layer, is_training, reuse, increase_dim=False, first=False):
    input_dim = input_layer.get_shape().as_list()[-1]

    if increase_dim:
        output_dim = input_dim * 2
        strides = (2,2)
    else:
        output_dim = input_dim
        strides = (1,1)

    with tf.variable_scope(scope):
        if first:
            h0 = input_layer
        else:
            h0_bn = batch_norm('h0_bn', input_layer, is_training, reuse)
            h0 = lrelu(h0_bn)

        h1_conv = conv2d('h1_conv', h0, output_dim, strides=strides)
        h1_bn = batch_norm('h1_bn', h1_conv, is_training, reuse)
        h1 = lrelu(h1_bn)

        h2_conv = conv2d('h2_conv', h1, output_dim)
        if increase_dim:
            l = avg_pool('l_pool', input_layer)
            l = tf.pad(l, [[0, 0], [0, 0],
                           [0, 0], [input_dim // 2, input_dim // 2]])
        else:
            l = input_layer
        h2 = tf.add(h2_conv, l)

        return h2