import tensorflow as tf

from DATA import *
from CONVNET import *

def conv_net_block(conv_net, net_info, tensor_list, is_first, is_training, act_o):
    seed = FLAGS['process_random_seed']
    trainable = conv_net['trainable']
    tensor = tensor_list[conv_net['input_index']]
    if is_first:
        layer_name_format = '%12s'
        net_info.architecture_log.append('========== net_name = %s ==========' % conv_net['net_name'])
        net_info.architecture_log.append('[%s][%4d] : (%s)' % (layer_name_format % 'input', tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list())))
        if FLAGS['mode_use_debug']:
            print(net_info.architecture_log[-2])
            print(net_info.architecture_log[-1])
    with tf.variable_scope(conv_net['net_name']):
        for l_index, layer_o in enumerate(conv_net['layers']):
            layer = layer_o['name']
            if layer == "relu":
                tensor = exe_relu_layer(tensor)
            elif layer == "prelu":
                tensor = exe_prelu_layer(tensor, net_info, l_index, is_first, act_o)
            elif layer == "lrelu":
                tensor = exe_lrelu_layer(tensor, layer_o)
            elif layer == "bn":
                tensor = exe_bn_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, act_o)
            elif layer == "in":
                tensor = exe_in_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "ln":
                tensor = exe_ln_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "conv":
                tensor = exe_conv_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, seed)
            elif layer == "conv_res":
                tensor = exe_conv_res_layer(tensor, layer_o, tensor_list, net_info, l_index, is_first, is_training, trainable, seed)
            elif layer == "res":
                tensor = exe_res_layer(tensor, layer_o, tensor_list)
            elif layer == "max_pool":
                tensor = exe_max_pool_layer(tensor, layer_o)
            elif layer == "avg_pool":
                tensor = exe_avg_pool_layer(tensor, layer_o)
            elif layer == "resize":
                tensor = exe_resize_layer(tensor, layer_o)
            elif layer == "concat":
                tensor = exe_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "g_concat":
                tensor = exe_global_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "reshape":
                tensor = exe_reshape_layer(tensor, layer_o)
            elif layer == "clip":
                tensor = exe_clip_layer(tensor, layer_o)
            elif layer == "sigmoid":
                tensor = exe_sigmoid_layer(tensor)
            elif layer == "softmax":
                tensor = exe_softmax_layer(tensor)
            elif layer == "squeeze":
                tensor = exe_squeeze_layer(tensor, layer_o)
            elif layer == "abs":
                tensor = exe_abs_layer(tensor)
            elif layer == "tanh":
                tensor = exe_tanh_layer(tensor)
            elif layer == "inv_tanh":
                tensor = exe_inv_tanh_layer(tensor)
            elif layer == "add":
                tensor = exe_add_layer(tensor, layer_o)
            elif layer == "mul":
                tensor = exe_mul_layer(tensor, layer_o)
            elif layer == "reduce_mean":
                tensor = exe_reduce_mean_layer(tensor, layer_o)
            elif layer == "null":
                tensor = exe_null_layer(tensor)
            elif layer == "selu":
                tensor = exe_selu_layer(tensor)
            else:
                assert False, 'Error layer name = %s' % layer
            tensor_list.append(tensor)

            if is_first:
                info = '[%s][%4d] : (%s)'% (layer_name_format % layer, tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list()))
                if 'index' in layer_o:
                    info = info + ', use index [%4d] : (%s)' % (layer_o['index'], ', '.join('%4d' % (-1 if v is None else v) for v in tensor_list[layer_o['index']].get_shape().as_list()))
                net_info.architecture_log.append(info)
                if FLAGS['mode_use_debug']:
                    print(info)

    return tensor

def model(net_info, tensor, is_training, act_o, is_first=False):
    tensor_list = [tensor]
    if net_info.name == "netD":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o)
        result = tensor_list[-1]
    elif net_info.name == "netG":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o)
        result = tensor_list[-1]
    else:
        assert False, 'net_info.name ERROR = %s' % net_info.name
    return result

def img_L2_loss(img1, img2, use_local_weight):
    if use_local_weight:
        w = -tf.log(tf.cast(img2, tf.float64) + tf.exp(tf.constant(-99, dtype=tf.float64))) + 1
        w = tf.cast(w * w, tf.float32)
        return tf.reduce_mean(w * tf.square(tf.sub(img1, img2)))
    else:
        return tf.reduce_mean(tf.square(tf.sub(img1, img2)))

def img_L1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(tf.sub(img1, img2)))

def img_GD_loss(img1, img2):
    img1 = tf_imgradient(tf.pack([img1]))
    img2 = tf_imgradient(tf.pack([img2]))
    return tf.reduce_mean(tf.square(tf.sub(img1, img2)))

def regularization_cost(net_info):
    cost = 0
    for w, p in zip(net_info.weights, net_info.parameter_names):
        if p[-2:] == "_w": 
            cost = cost + (tf.nn.l2_loss(w))
    return cost