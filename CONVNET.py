import tensorflow as tf

PARAMETERS_NAME = ["conv_%d_w", \
                   "conv_%d_b", \
                   "prelu_%d_%d_alpha", \
                   "bn_%d_%d_offset", \
                   "bn_%d_%d_scale", \
                   "bn_%d_%d_mv_mean", \
                   "bn_%d_%d_mv_var", \
                   "in_%d_%d_offset", \
                   "in_%d_%d_scale", \
                   "ln_%d_%d_offset", \
                   "ln_%d_%d_scale"]

#  .#####...######..##......##..##.
#  .##..##..##......##......##..##.
#  .#####...####....##......##..##.
#  .##..##..##......##......##..##.
#  .##..##..######..######...####..
#  ................................
def relu_layer():
    return dict(name='relu')

def exe_relu_layer(tensor):
    tensor = tf.nn.relu(tensor)
    return tensor

#  .#####...#####...######..##......##..##.
#  .##..##..##..##..##......##......##..##.
#  .#####...#####...####....##......##..##.
#  .##......##..##..##......##......##..##.
#  .##......##..##..######..######...####..
#  ........................................
def prelu_layer():
    return dict(name='prelu')

def exe_prelu_layer(tensor, net_info, l_index, is_first, act_o):
    p_index = 2
    parameter_count = 1
    alphas_l = []
    for i in range(act_o['size']):
        alphas = tf.get_variable(name=PARAMETERS_NAME[p_index] % (l_index, i), \
                                 shape=tensor.get_shape()[-1], \
                                 initializer=tf.constant_initializer(0.0))
        alphas_l.append(alphas)
    alphas = alphas_l[act_o['index']]
    pos = tf.nn.relu(tensor)
    neg = alphas * (tensor - abs(tensor)) * 0.5
    tensor = pos + neg
    if is_first:
        net_info.weights.extend(alphas_l)
        for i in range(parameter_count):
            for j in range(act_o['size']):
                net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % (l_index, j))
    return tensor

#  .##......#####...######..##......##..##.
#  .##......##..##..##......##......##..##.
#  .##......#####...####....##......##..##.
#  .##......##..##..##......##......##..##.
#  .######..##..##..######..######...####..
#  ........................................
def lrelu_layer(leak):
    return dict(
        name='lrelu',
        leak=leak)

def exe_lrelu_layer(tensor, layer_o):
    leak = layer_o['leak']
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    tensor = f1 * tensor + f2 * abs(tensor)
    return tensor

#  ..####...######..##......##..##.
#  .##......##......##......##..##.
#  ..####...####....##......##..##.
#  .....##..##......##......##..##.
#  ..####...######..######...####..
#  ................................
def selu_layer():
    return dict(name='selu')

def exe_selu_layer(tensor):
    #alpha = 1.6732632423543772848170429916717
    #scale = 1.0507009873554804934193349852946
    alpha, scale = (1.0198755295894968, 1.0026538655307724)
    return scale*tf.where(tensor>=0.0, tensor, alpha*tf.nn.elu(tensor))
    
#  .#####...##..##.
#  .##..##..###.##.
#  .#####...##.###.
#  .##..##..##..##.
#  .#####...##..##.
#  ................
def bn_layer(use_offset=False, use_scale=False, epsilon=1e-5, decay=0.9):
    return dict(
        name='bn',
        use_offset=use_offset,
        use_scale=use_scale,
        epsilon=epsilon,
        decay=decay)

def exe_bn_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, act_o):
    p_index = 3
    parameter_count = 4
    shape = [tensor.get_shape()[-1]]
    offset_trainable = layer_o['use_offset'] if trainable else False
    scale_trainable  = layer_o['use_scale']  if trainable else False

    pars = []
    for i in range(act_o['size']):
        offset  = tf.get_variable(name=PARAMETERS_NAME[p_index  ] % (l_index, i), shape=shape, initializer=tf.constant_initializer(0.0), trainable=offset_trainable)
        scale   = tf.get_variable(name=PARAMETERS_NAME[p_index+1] % (l_index, i), shape=shape, initializer=tf.constant_initializer(1.0), trainable=scale_trainable)
        mv_mean = tf.get_variable(name=PARAMETERS_NAME[p_index+2] % (l_index, i), shape=shape, initializer=tf.constant_initializer(0.0), trainable=False)
        mv_var  = tf.get_variable(name=PARAMETERS_NAME[p_index+3] % (l_index, i), shape=shape, initializer=tf.constant_initializer(1.0), trainable=False)
        pars.append([offset, scale, mv_mean, mv_var])
    offset, scale, mv_mean, mv_var = pars[act_o['index']]

    if is_first:
        for ps in pars:
            net_info.weights.extend(ps)
        for i in range(parameter_count):
            for j in range(act_o['size']):
                net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % (l_index, j))
    if is_training:
        batch_mean, batch_var = tf.nn.moments(tensor, [0, 1, 2])
        train_mean = tf.assign(mv_mean,
                               mv_mean * layer_o['decay'] + batch_mean * (1 - layer_o['decay']))
        train_var  = tf.assign(mv_var,
                               mv_var  * layer_o['decay'] + batch_var  * (1 - layer_o['decay']))
        with tf.control_dependencies([train_mean, train_var]):
            tensor = tf.nn.batch_normalization(tensor, batch_mean, batch_var, offset, scale, layer_o['epsilon'])
    else:
        tensor = tf.nn.batch_normalization(tensor, mv_mean, mv_var, offset, scale, layer_o['epsilon'])

    return tensor

#  .######..##..##.
#  ...##....###.##.
#  ...##....##.###.
#  ...##....##..##.
#  .######..##..##.
#  ................
def in_layer(use_offset=False, use_scale=False, epsilon=1e-5):
    return dict(
        name='in',
        use_offset=use_offset,
        use_scale=use_scale,
        epsilon=epsilon)

def exe_in_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o):
    p_index = 7
    shape = [tensor.get_shape()[-1]]
    offset_trainable = layer_o['use_offset'] if trainable else False
    scale_trainable  = layer_o['use_scale']  if trainable else False
    pars = []
    for i in range(act_o['size']):
        offset = tf.get_variable(name=PARAMETERS_NAME[p_index  ] % (l_index, i), shape=shape, initializer=tf.constant_initializer(0.0), trainable=offset_trainable)
        scale  = tf.get_variable(name=PARAMETERS_NAME[p_index+1] % (l_index, i), shape=shape, initializer=tf.constant_initializer(1.0), trainable=scale_trainable)
        pars.append([offset, scale])
    offset, scale = pars[act_o['index']]

    if is_first:
        for ps in pars:
            net_info.weights.extend(ps)
        parameter_count = 2
        for i in range(parameter_count):
            for j in range(act_o['size']):
                net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % (l_index, j))

    t_list = tf.unpack(tensor)
    result = []
    for t in t_list:
        batch_mean, batch_var = tf.nn.moments(t, [0, 1])
        t = tf.nn.batch_normalization(t, batch_mean, batch_var, offset, scale, layer_o['epsilon'])
        result.append(t)
    return tf.pack(result)

    # mean, variance = tf.nn.moments(tensor, axes=[1,2], keep_dims=True)
    # epsilon = layer_o['epsilon']
    # inv = tf.rsqrt(variance + epsilon)
    # normalized = (tensor - mean)*inv
    # return scale*normalized + offset
    
    # mean, var = tf.nn.moments(tensor, [1, 2], keep_dims=True)
    # normalized = tf.div(tf.sub(tensor, mean), tf.sqrt(tf.add(var, layer_o['epsilon'])))
    # return scale * normalized + offset

#  .##......##..##.
#  .##......###.##.
#  .##......##.###.
#  .##......##..##.
#  .######..##..##.
#  ................
def ln_layer(use_offset=False, use_scale=False, epsilon=1e-5):
    return dict(
        name='ln',
        use_offset=use_offset,
        use_scale=use_scale,
        epsilon=epsilon)

def exe_ln_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o):
    p_index = 9
    shape = [1, 1, tensor.get_shape()[-1]]
    offset_trainable = layer_o['use_offset'] if trainable else False
    scale_trainable  = layer_o['use_scale']  if trainable else False
    pars = []
    for i in range(act_o['size']):
        offset = tf.get_variable(name=PARAMETERS_NAME[p_index  ] % (l_index, i), shape=shape, initializer=tf.constant_initializer(0.0), trainable=offset_trainable)
        scale  = tf.get_variable(name=PARAMETERS_NAME[p_index+1] % (l_index, i), shape=shape, initializer=tf.constant_initializer(1.0), trainable=scale_trainable)
        pars.append([offset, scale])
    offset, scale = pars[act_o['index']]

    if is_first:
        for ps in pars:
            net_info.weights.extend(ps)
        parameter_count = 2
        for i in range(parameter_count):
            for j in range(act_o['size']):
                net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % (l_index, j))
    mean, var = tf.nn.moments(tensor, [1, 2, 3], keep_dims=True)
    result = tf.nn.batch_normalization(tensor, mean, var, offset, scale, layer_o['epsilon'])
    return result

#  ..####....####...##..##..##..##.
#  .##..##..##..##..###.##..##..##.
#  .##......##..##..##.###..##..##.
#  .##..##..##..##..##..##...####..
#  ..####....####...##..##....##...
#  ................................
def conv_layer(kernel, stride, filter, pad_mode, initializer, dropout=1, padding='VALID'):
    return dict(
        name='conv',
        kernel=kernel,
        stride=stride,
        filter=filter,
        pad_mode=pad_mode,
        initializer=initializer,
        dropout=dropout,
        padding=padding)

def exe_conv_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, seed):
    p_index = 0
    parameter_count = 2
    kernel = layer_o['kernel']
    stride = layer_o['stride']
    filter = layer_o['filter']
    pad_mode = layer_o['pad_mode']
    dropout = layer_o['dropout']
    initializer = layer_o['initializer']
    padding = layer_o['padding']
    conv_w = tf.get_variable(PARAMETERS_NAME[p_index  ] % l_index, \
                            [kernel, kernel, tensor.get_shape()[-1], filter], \
                            initializer=initializer, \
                            trainable=trainable)
    conv_b = tf.get_variable(PARAMETERS_NAME[p_index+1] % l_index, \
                            [filter], \
                            initializer=tf.constant_initializer(0), \
                            trainable=trainable)
    pad_size = (kernel - 1) // 2
    if pad_size > 0 and pad_mode is not None:
        tensor = tf.pad(tensor, [[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]], pad_mode)
    if is_training and dropout < 1:
        tensor = tf.nn.dropout(tensor, dropout, seed=seed)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,stride,stride,1], padding=padding), conv_b)
    if is_first:
        net_info.weights.extend((conv_w, conv_b))
        for i in range(parameter_count):
            net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % l_index)
    return tensor

#  ..####....####...##..##..##..##..........#####...######...####...######..#####...##..##...####...##.....
#  .##..##..##..##..###.##..##..##..........##..##..##......##........##....##..##..##..##..##..##..##.....
#  .##......##..##..##.###..##..##..........#####...####.....####.....##....##..##..##..##..######..##.....
#  .##..##..##..##..##..##...####...........##..##..##..........##....##....##..##..##..##..##..##..##.....
#  ..####....####...##..##....##....######..##..##..######...####...######..#####....####...##..##..######.
#  ........................................................................................................
def conv_res_layer(index, kernel, stride, initializer, dropout=1, padding='VALID'):
    return dict(
        name='conv_res',
        index=index,
        kernel=kernel,
        stride=stride,
        dropout=dropout,
        initializer=initializer,
        padding=padding)

def exe_conv_res_layer(res_tensor, layer_o, tensor_list, net_info, l_index, is_first, is_training, trainable, seed):
    p_index = 0
    parameter_count = 2
    index = layer_o['index']
    kernel = layer_o['kernel']
    stride = layer_o['stride']
    dropout = layer_o['dropout']
    initializer = layer_o['initializer']
    padding = layer_o['padding']
    filter = res_tensor.get_shape()[-1]
    tensor = tensor_list[index]
    conv_w = tf.get_variable(PARAMETERS_NAME[p_index  ] % l_index, \
                            [kernel, kernel, tensor.get_shape()[-1], filter], \
                            initializer=initializer, \
                            trainable=trainable)
    conv_b = tf.get_variable(PARAMETERS_NAME[p_index+1] % l_index, \
                            [filter], \
                            initializer=tf.constant_initializer(0), \
                            trainable=trainable)
    if is_training and dropout < 1:
        tensor = tf.nn.dropout(tensor, dropout, seed=seed)
    tensor = tf.nn.bias_add(tf.nn.conv2d(tensor, conv_w, strides=[1,stride,stride,1], padding=padding), conv_b)
    if is_first:
        net_info.weights.extend((conv_w, conv_b))
        for i in range(parameter_count):
            net_info.parameter_names.append(PARAMETERS_NAME[p_index + i] % l_index)
    tensor = tf.add(res_tensor, tensor)
    return tensor

#  .#####...######...####...######..#####...##..##...####...##.....
#  .##..##..##......##........##....##..##..##..##..##..##..##.....
#  .#####...####.....####.....##....##..##..##..##..######..##.....
#  .##..##..##..........##....##....##..##..##..##..##..##..##.....
#  .##..##..######...####...######..#####....####...##..##..######.
#  ................................................................
def res_layer(index, axis=None):
    return dict(
        name='res',
        index=index,
        axis=axis)

def exe_res_layer(tensor, layer_o, tensor_list):
    index = layer_o['index']
    axis = layer_o['axis']
    res_tensor = tensor_list[index]
    if axis is not None:
        l = [res_tensor[:, :, :, i] for i in axis]
        res_tensor = tf.pack(l, -1)
    tensor = tf.add(tensor, res_tensor)
    return tensor

#  .##...##...####...##..##..........#####....####....####...##.....
#  .###.###..##..##...####...........##..##..##..##..##..##..##.....
#  .##.#.##..######....##............#####...##..##..##..##..##.....
#  .##...##..##..##...####...........##......##..##..##..##..##.....
#  .##...##..##..##..##..##..######..##.......####....####...######.
#  .................................................................
def max_pool_layer(kernel, stride, padding='VALID'):
    return dict(
        name='max_pool',
        kernel=kernel,
        stride=stride,
        padding=padding)

def exe_max_pool_layer(tensor, layer_o):
    kernel = layer_o['kernel']
    stride = layer_o['stride']
    padding = layer_o['padding']
    tensor = tf.nn.max_pool(tensor, [1, kernel, kernel, 1], [1, stride, stride, 1], padding=padding)
    return tensor

#  ..####...##..##...####...........#####....####....####...##.....
#  .##..##..##..##..##..............##..##..##..##..##..##..##.....
#  .######..##..##..##.###..........#####...##..##..##..##..##.....
#  .##..##...####...##..##..........##......##..##..##..##..##.....
#  .##..##....##.....####...######..##.......####....####...######.
#  ................................................................
def avg_pool_layer(kernel, stride, padding='VALID'):
    return dict(
        name='avg_pool',
        kernel=kernel,
        stride=stride,
        padding=padding)

def exe_avg_pool_layer(tensor, layer_o):
    kernel = layer_o['kernel']
    stride = layer_o['stride']
    padding = layer_o['padding']
    tensor = tf.nn.avg_pool(tensor, [1, kernel, kernel, 1], [1, stride, stride, 1], padding=padding)
    return tensor

#  .#####...######...####...######..######..######.
#  .##..##..##......##........##.......##...##.....
#  .#####...####.....####.....##......##....####...
#  .##..##..##..........##....##.....##.....##.....
#  .##..##..######...####...######..######..######.
#  ................................................
def resize_layer(scale, method, align_corners=False):
    return dict(
        name='resize',
        scale=scale,
        method=method,
        align_corners=align_corners)

def exe_resize_layer(tensor, layer_o):
    scale = layer_o['scale']
    method = layer_o['method']
    align_corners = layer_o['align_corners']
    t_shape = tensor.get_shape().as_list()
    if t_shape[1] == None or t_shape[2] == None:
        t_shape = tf.shape(tensor)
    t_size = [t_shape[1] * scale, t_shape[2] * scale]
    tensor = tf.image.resize_images(tensor, t_size, method=method, align_corners=align_corners)
    return tensor

#  ..####....####...##..##...####....####...######.
#  .##..##..##..##..###.##..##..##..##..##....##...
#  .##......##..##..##.###..##......######....##...
#  .##..##..##..##..##..##..##..##..##..##....##...
#  ..####....####...##..##...####...##..##....##...
#  ................................................
def concat_layer(index):
    return dict(
        name='concat',
        index=index)

def exe_concat_layer(tensor, layer_o, tensor_list):
    index = layer_o['index']
    concat_t = tensor_list[index]
    tensor = tf.concat(3, [tensor, concat_t])
    return tensor

#  ..####...##.......####...#####....####...##...............####....####...##..##...####....####...######.
#  .##......##......##..##..##..##..##..##..##..............##..##..##..##..###.##..##..##..##..##....##...
#  .##.###..##......##..##..#####...######..##..............##......##..##..##.###..##......######....##...
#  .##..##..##......##..##..##..##..##..##..##..............##..##..##..##..##..##..##..##..##..##....##...
#  ..####...######...####...#####...##..##..######..######...####....####...##..##...####...##..##....##...
#  ........................................................................................................
def global_concat_layer(index):
    return dict(
        name='g_concat',
        index=index)

def exe_global_concat_layer(tensor, layer_o, tensor_list):
    index = layer_o['index']
    h = tf.shape(tensor)[1]
    w = tf.shape(tensor)[2]
    concat_t = tf.squeeze(tensor_list[index], [1, 2])
    dims = concat_t.get_shape()[-1]
    batch_l = tf.unpack(concat_t, axis=0)
    bs = []
    for batch in batch_l:
        batch = tf.tile(batch, [h * w])
        batch = tf.reshape(batch, [h, w, -1])
        bs.append(batch)
    concat_t = tf.pack(bs)
    concat_t.set_shape(concat_t.get_shape().as_list()[:3] + [dims])
    tensor = tf.concat(3, [tensor, concat_t])
    return tensor

#  .#####...######...####...##..##...####...#####...######.
#  .##..##..##......##......##..##..##..##..##..##..##.....
#  .#####...####.....####...######..######..#####...####...
#  .##..##..##..........##..##..##..##..##..##......##.....
#  .##..##..######...####...##..##..##..##..##......######.
#  ........................................................
def reshape_layer(shape):
    return dict(
        name='reshape',
        shape=shape)

def exe_reshape_layer(tensor, layer_o):
    shape = reshape['shape']
    shape = [tensor.get_shape().as_list()[0]] + shape
    tensor = tf.reshape(tensor, shape)
    return tensor

#  ..####...##......######..#####..
#  .##..##..##........##....##..##.
#  .##......##........##....#####..
#  .##..##..##........##....##.....
#  ..####...######..######..##.....
#  ................................
def clip_layer(min_v=0, max_v=1):
    return dict(
        name='clip',
        min_v=min_v,
        max_v=max_v)

def exe_clip_layer(tensor, layer_o):
    min_v = layer_o['min_v']
    max_v = layer_o['max_v']
    tensor = tf.clip_by_value(tensor, min_v, max_v)
    return tensor

#  ..####...######...####...##...##...####...######..#####..
#  .##........##....##......###.###..##..##....##....##..##.
#  ..####.....##....##.###..##.#.##..##..##....##....##..##.
#  .....##....##....##..##..##...##..##..##....##....##..##.
#  ..####...######...####...##...##...####...######..#####..
#  .........................................................

def sigmoid_layer():
    return dict(name='sigmoid')

def exe_sigmoid_layer(tensor):
    return tf.nn.sigmoid(tensor)

#  ..####....####...######..######..##...##...####...##..##.
#  .##......##..##..##........##....###.###..##..##...####..
#  ..####...##..##..####......##....##.#.##..######....##...
#  .....##..##..##..##........##....##...##..##..##...####..
#  ..####....####...##........##....##...##..##..##..##..##.
#  .........................................................
def softmax_layer():
    return dict(name='softmax')

def exe_softmax_layer(tensor):
    return tf.nn.softmax(tensor)


#  ..####....####...##..##..######..######..######..######.
#  .##......##..##..##..##..##......##.........##...##.....
#  ..####...##.###..##..##..####....####......##....####...
#  .....##..##..##..##..##..##......##.......##.....##.....
#  ..####....#####...####...######..######..######..######.
#  ........................................................
def squeeze_layer(axis):
    return dict(name='squeeze', axis=axis)

def exe_squeeze_layer(tensor, layer_o):
    axis = layer_o['axis']
    return tf.squeeze(tensor, axis)

#  ..####...#####....####..
#  .##..##..##..##..##.....
#  .######..#####....####..
#  .##..##..##..##......##.
#  .##..##..#####....####..
#  ........................
def abs_layer():
    return dict(name='abs')

def exe_abs_layer(tensor):
    return tf.abs(tensor)

#  .######...####...##..##..##..##.
#  ...##....##..##..###.##..##..##.
#  ...##....######..##.###..######.
#  ...##....##..##..##..##..##..##.
#  ...##....##..##..##..##..##..##.
#  ................................
def tanh_layer():
    return dict(name='tanh')

def exe_tanh_layer(tensor):
    return tf.tanh(tensor)

#  .######..##..##..##..##..........######...####...##..##..##..##.
#  ...##....###.##..##..##............##....##..##..###.##..##..##.
#  ...##....##.###..##..##............##....######..##.###..######.
#  ...##....##..##...####.............##....##..##..##..##..##..##.
#  .######..##..##....##....######....##....##..##..##..##..##..##.
#  ................................................................
def inv_tanh_layer():
    return dict(name='inv_tanh')

def exe_inv_tanh_layer(tensor):
    return -tf.log((2.0 / (tensor + 1 + 1e-100)) - 1) * 0.5

#  ..####...#####...#####..
#  .##..##..##..##..##..##.
#  .######..##..##..##..##.
#  .##..##..##..##..##..##.
#  .##..##..#####...#####..
#  ........................
def add_layer(value):
    return dict(name='add', value=value)

def exe_add_layer(tensor, layer_o):
    value = layer_o['value']
    return tf.add(tensor, value)

#  .##...##..##..##..##.....
#  .###.###..##..##..##.....
#  .##.#.##..##..##..##.....
#  .##...##..##..##..##.....
#  .##...##...####...######.
#  .........................
def mul_layer(value):
    return dict(name='mul', value=value)

def exe_mul_layer(tensor, layer_o):
    value = layer_o['value']
    return tf.mul(tensor, value)

#  .##..##..##..##..##......##.....
#  .###.##..##..##..##......##.....
#  .##.###..##..##..##......##.....
#  .##..##..##..##..##......##.....
#  .##..##...####...######..######.
#  ................................
def null_layer():
    return dict(name='null')

def exe_null_layer(tensor):
    return tensor

#  .#####...######..#####...##..##...####...######..........##...##..######...####...##..##.
#  .##..##..##......##..##..##..##..##..##..##..............###.###..##......##..##..###.##.
#  .#####...####....##..##..##..##..##......####............##.#.##..####....######..##.###.
#  .##..##..##......##..##..##..##..##..##..##..............##...##..##......##..##..##..##.
#  .##..##..######..#####....####....####...######..######..##...##..######..##..##..##..##.
#  .........................................................................................
def reduce_mean_layer(axis=None, keep_dims=False):
    return dict(name='reduce_mean', axis=axis, keep_dims=keep_dims)

def exe_reduce_mean_layer(tensor, layer_o):
    axis = layer_o['axis']
    keep_dims = layer_o['keep_dims']
    return tf.reduce_mean(tensor, axis, keep_dims)

#  .#####...######...####...######..#####...##..##...####...##..............#####...##.......####....####...##..##.
#  .##..##..##......##........##....##..##..##..##..##..##..##..............##..##..##......##..##..##..##..##.##..
#  .#####...####.....####.....##....##..##..##..##..######..##..............#####...##......##..##..##......####...
#  .##..##..##..........##....##....##..##..##..##..##..##..##..............##..##..##......##..##..##..##..##.##..
#  .##..##..######...####...######..#####....####...##..##..######..######..#####...######...####....####...##..##.
#  ................................................................................................................
def residual_block(input_p, output_p, stride, initializer, index):
    result = []
    bottle_p = output_p // 4
    result.append(bn_layer(True, True))
    result.append(prelu_layer())
    result.append(conv_layer(1, stride, bottle_p, None, initializer))
    result.append(bn_layer(True, True))
    result.append(prelu_layer())
    result.append(conv_layer(3, 1, bottle_p, "CONSTANT", initializer))
    result.append(bn_layer(True, True))
    result.append(prelu_layer())
    result.append(conv_layer(1, 1, output_p, None, initializer))
    if input_p == output_p:
        result.append(res_layer(index))
    else:
        result.append(conv_res_layer(index + 2, 1, stride, initializer))
    return result

def residual_layer(count, input_p, output_p, stride, initializer, index):
    result = residual_block(input_p, output_p, stride, initializer, index)
    for _ in range(count - 1):
        index = index + 10
        result = result + residual_block(output_p, output_p, 1, initializer, index)
    return result
