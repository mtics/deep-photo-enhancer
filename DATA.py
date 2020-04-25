import tensorflow as tf
import numpy as np
import random, os

from CONVNET import *

# Configure
FLAGS = {}
FLAGS['num_gpu'] = '3'
FLAGS['num_exp'] = 736
FLAGS['num_epoch'] = 90
FLAGS['method'] = 'Unsupervised on HDR dataset (627)'
FLAGS['mode_use_debug'] = False
FLAGS['netG_init_method'] = 'var_scale'  # var_scale, rand_uniform, rand_normal, truncated_normal
FLAGS['netG_init_weight'] = 1e-3
FLAGS['netG_base_learning_rate'] = 1e-5
FLAGS['format_log_step'] = '%.3f'
FLAGS['root_path'] = os.path.dirname(__file__)
FLAGS['load_model_path'] = FLAGS['root_path'] + '/model/' + '%s.ckpt' % (FLAGS['format_log_step'] % FLAGS['num_epoch'])
FLAGS['load_model_path_new'] = FLAGS['root_path'] + '/model/' + '%s-new.ckpt' % (
            FLAGS['format_log_step'] % FLAGS['num_epoch'])
# FLAGS['load_model_path_new'] = 'model/' + '%s-new.ckpt' % (
#             FLAGS['format_log_step'] % FLAGS['num_epoch'])
FLAGS['data_output_ext'] = '.png'
FLAGS['data_input_dtype'] = np.uint8
FLAGS['data_compute_dtype'] = np.float32
FLAGS['data_image_size'] = 512
FLAGS['data_image_channel'] = 3
FLAGS['process_random_seed'] = 2
FLAGS['folder_input'] = '../static/uploads/'
FLAGS['folder_test_img'] = '../static/result/'
random.seed(FLAGS['process_random_seed'])


class DataFlowMat(object):
    def __init__(self, b):
        self.rect = [tf.placeholder(tf.int32) for _ in range(b)]
        self.rot = [tf.placeholder(tf.int32) for _ in range(b)]


class DataFlow(object):
    def __init__(self):
        b = 1
        self.input1_src = tf.placeholder(tf.as_dtype(FLAGS['data_input_dtype']),
                                         shape=[b, FLAGS['data_image_size'], FLAGS['data_image_size'],
                                                FLAGS['data_image_channel']])
        self.input1 = tf.cast(self.input1_src, FLAGS['data_compute_dtype']) / self.input1_src.dtype.max
        self.mat1 = DataFlowMat(b)


def flatten_list(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten_list(x))
    else:
        result.append(xs)
    return result


class NetInfo(object):
    def __init__(self, name):
        self.CONV_NETS = []
        seed = FLAGS['process_random_seed']
        ich = FLAGS['data_image_channel']
        if name[:4] == "netG":
            init_w = FLAGS['netG_init_weight']
            if FLAGS['netG_init_method'] == "var_scale":
                initializer = tf.contrib.layers.variance_scaling_initializer(init_w, seed=seed)
            elif FLAGS['netG_init_method'] == "rand_uniform":
                initializer = tf.random_uniform_initializer(-init_w * np.sqrt(3), init_w * np.sqrt(3), seed=seed)
            elif FLAGS['netG_init_method'] == "rand_normal":
                initializer = tf.random_normal_initializer(mean=0., stddev=init_w, seed=seed)
            elif FLAGS['netG_init_method'] == "truncated_normal":
                initializer = tf.truncated_normal_initializer(mean=0., stddev=init_w, seed=seed)
            nonlinearity = selu_layer()  # prelu_layer()
            norm = bn_layer(True, True)
            act = [nonlinearity, norm]
            net_1 = dict(net_name='%s_1' % name, trainable=True)
            net_1['input_index'] = 0
            net_1['layers'] = flatten_list([ \
                conv_layer(3, 1, 16, "SYMMETRIC", initializer), act, \
                conv_layer(5, 2, 32, "SYMMETRIC", initializer), act, \
                conv_layer(5, 2, 64, "SYMMETRIC", initializer), act, \
                conv_layer(5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer(5, 2, 128, "SYMMETRIC", initializer), act, \
                ])
            self.CONV_NETS.append(net_1)

            net_2 = dict(net_name='%s_2' % name, trainable=True)
            net_2['input_index'] = 15
            net_2['layers'] = flatten_list([ \
                conv_layer(5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer(5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer(8, 1, 128, None, initializer), nonlinearity, \
                conv_layer(1, 1, 128, None, initializer) \
                ])
            self.CONV_NETS.append(net_2)
            net_3 = dict(net_name='%s_3' % name, trainable=True)
            net_3['input_index'] = 15
            net_3['layers'] = flatten_list([ \
                conv_layer(3, 1, 128, "SYMMETRIC", initializer), global_concat_layer(24), \
                conv_layer(1, 1, 128, "SYMMETRIC", initializer), act, \
                conv_layer(3, 1, 128, "SYMMETRIC", initializer),
                resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer(10), act, \
                conv_layer(3, 1, 128, "SYMMETRIC", initializer),
                resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer(7), act, \
                conv_layer(3, 1, 64, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                concat_layer(4), act, \
                conv_layer(3, 1, 32, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                concat_layer(1), act, \
                conv_layer(3, 1, 16, "SYMMETRIC", initializer), act, \
                conv_layer(3, 1, ich, "SYMMETRIC", initializer), res_layer(0, [0, 1, 2]) \
                # , clip_layer() \
            ])
            self.CONV_NETS.append(net_3)
        else:
            assert False, 'net name error'

        self.architecture_log = []
        self.weights = []
        self.parameter_names = []
        self.name = name
        self.variable_scope_name = name + '_var_scope'
