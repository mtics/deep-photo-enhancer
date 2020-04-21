import tensorflow as tf
import numpy as np
import os, cv2, random

from time import localtime, strftime
from datetime import datetime
from scipy import io

class ImagePool:
  """ History of generated images
      Same logic as https://github.com/junyanz/CycleGAN/blob/master/util/image_pool.lua
  """
  def __init__(self, pool_size):
    self.pool_size = pool_size
    self.images = []

  def query(self, image):
    if self.pool_size == 0:
      return image

    if len(self.images) < self.pool_size:
      self.images.append(image)
      return image
    else:
      p = random.random()
      if p > 0.5:
        # use old image
        random_id = random.randrange(0, self.pool_size)
        tmp = self.images[random_id].copy()
        self.images[random_id] = image.copy()
        return tmp
      else:
        return image
        
def current_time():
    return strftime("%Y-%m-%d %H:%M:%S", localtime())

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self) :
        for f in self.files:
            f.flush()

class Timer(object):
    def zero_time(self):
        return datetime.min - datetime.min
    def start(self):
        self.start_time = datetime.now()
    def end(self):
        return datetime.now() - self.start_time

def make_dirs(dir_list):
    for folder in dir_list:
        if not os.path.exists(folder):
            os.makedirs(folder)

def get_receptive_field(kernels):
    shave = 0
    for i, k in enumerate(kernels):
        shave = shave + (k - 1) // 2
    return shave

def read_file_to_list(file, transfer_type, key_value=True):
    with open(file, 'r') as f:
        if key_value:
            datas = [line.strip().split('\t') for line in f]
            result = [transfer_type(data[1]) for data in datas]
        else:
            result = [transfer_type(line.rstrip('\n')) for line in f]
    return result

def get_file_list(abs_path, ext):
    result = []
    for file in os.listdir(abs_path):
        if file.endswith("." + ext):
            result.append(os.path.join(abs_path, file))
    return result

def write_list_to_file(file, data_list, name_list):
    with open(file, 'w') as f:
        assert(len(data_list) == len(name_list))
        n = len(data_list) // len(name_list)
        for i, s in enumerate(data_list):
            f.write(name_list[i // n] + '\t' + repr(s) + '\n')

def insert_string_to_list(index, string, data_list):
    for i in range(len(data_list)):
        if index == -1:
            ind = len(data_list[i])
        else:
            ind = index    
        data_list[i] = data_list[i][:ind] + string + data_list[i][ind:]
    return data_list

def save_net(parameter_names, path):
    net_list = np.array(parameter_names, dtype=np.object)
    io.savemat(path, mdict={'net': net_list})

def save_weights(weights_data, parameter_names, path, now_epoch):
    weights_dict = {}
    for i in range(len(weights_data)):
        weights_dict[parameter_names[i]] = weights_data[i]
    io.savemat(path + now_epoch, weights_dict)

def save_model(saver, sess, path, now_epoch):
    model_save_path = path + now_epoch + ".ckpt"
    saver.save(sess, save_path=model_save_path)
    print(current_time() + ", =============================================================== Model saved in file: %s" % (now_epoch + ".ckpt"))

def calculate_rect(img, angle, scale):
    R = cv2.getRotationMatrix2D((img.shape[1]/2.0, img.shape[0]/2.0), angle, scale)
    corners = np.zeros((3,4))
    corners[0,0] = 0
    corners[0,1] = img.shape[1]
    corners[0,2] = 0
    corners[0,3] = img.shape[1]
    corners[1,0] = 0
    corners[1,1] = 0
    corners[1,2] = img.shape[0]
    corners[1,3] = img.shape[0]
    corners[2:] = 1

    c = np.dot(R, corners)

    x = c[0,0]
    y = c[1,0]

    left = x
    right = x
    up = y
    down = y

    for i in range(4):
        x = c[0,i]
        y = c[1,i]
        if (x < left): left = x
        if (x > right): right = x
        if (y < up): up = y
        if (y > down): down = y
    h = down - up
    w = right - left
    return h, w, up, left

def rotate_image(img, angle, pad_for_enhancer):
    h, w, up, left = calculate_rect(img, angle, 1)
    h = int(round(h))
    w = int(round(w))
    R = cv2.getRotationMatrix2D((img.shape[1]/2.0, img.shape[0]/2.0), angle, 1)
    R[0, 2] = R[0, 2] - left
    R[1, 2] = R[1, 2] - up
    img = cv2.warpAffine(img, R, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE if pad_for_enhancer else cv2.BORDER_CONSTANT)
    return img

def data_augmentation(data, number, max_da, size, pad_for_enhancer, use_random):
    if not isinstance(data, np.ndarray):
        return data
    angle = (360 * 2) // max_da
    index = number // 2
    output = data if number % 2 == 0 else np.fliplr(data)
    if max_da == 8:
        output = np.rot90(output, index)
        mask = np.ones(shape=output.shape, dtype=np.float32)
    else:
        mask = np.ones(shape=output.shape, dtype=np.float32)
        output = rotate_image(output, index * angle, pad_for_enhancer)
        h = output.shape[0]
        w = output.shape[1]
        if h > w:
            w = int(round(w * size / h))
            h = size
        else:
            h = int(round(h * size / w))
            w = size
        output = cv2.resize(output, (w, h), interpolation=cv2.INTER_AREA)

        mask = rotate_image(mask, index * angle, False)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_AREA)
        mask = np.round(mask)
        mask = np.clip(mask, 0, 1)

    output, mask, rect = random_pad_to_size(output, size, mask, pad_for_enhancer, use_random)
    return output, mask, rect

def safe_casting(data, dtype):
    output = np.clip(data + 0.5, np.iinfo(dtype).min, np.iinfo(dtype).max)
    output = output.astype(dtype)
    return output

def abs_mean_of_list(x):
    list_mean = [np.mean(np.fabs(np.array(l))) for l in x]
    return sum(list_mean) / len(list_mean)

def gcd(a,b):
    while b > 0:
        a, b = b, a % b
    return a
    
def lcm(a, b):
    return a * b // gcd(a, b)

def random_pad_to_size(img, size, mask, pad_symmetric, use_random):
    if mask is None:
        mask = np.ones(shape=img.shape)
    s0 = size - img.shape[0]
    s1 = size - img.shape[1]

    if use_random:
        b0 = np.random.randint(0, s0 + 1)
        b1 = np.random.randint(0, s1 + 1)
    else:
        b0 = 0
        b1 = 0
    a0 = s0 - b0
    a1 = s1 - b1
    if pad_symmetric:
        img  = np.pad(img,  ((b0, a0), (b1, a1), (0, 0)), 'symmetric')
    else:
        img  = np.pad(img,  ((b0, a0), (b1, a1), (0, 0)), 'constant')
    mask = np.pad(mask, ((b0, a0), (b1, a1), (0, 0)), 'constant')
    return img, mask, [b0, img.shape[0] - a0, b1, img.shape[1] - a1]

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    g = np.exp(-((x*x + y*y)/(2.0*sigma*sigma)))
    g = np.expand_dims(g, -1)
    g = g.astype(np.float32)
    return g / np.sum(g)

def normalize_to_one_score(scores):
    s = 0
    score_weight = [1, 2, 3, 4, 5]
    for i, sw in enumerate(score_weight):
        s = s + sw * scores[i]
    scores = (s - 1) / 4.0
    return scores
    
def tf_normlize_to_one_score(scores):
    result = []
    score_weight = [1, 2, 3, 4, 5]
    for i, s in enumerate(score_weight):
        result.append(scores[:, i] * s)
    result = tf.reduce_sum(tf.pack(result, -1), axis=1, keep_dims=True)
    return result

def tf_accumulate(tensor):
    batch_list = tf.unpack(tensor)
    result = []
    for t in batch_list:
        channel_list = tf.unpack(t)
        for c in range(1, len(channel_list)):
            channel_list[c] = channel_list[c] + channel_list[c-1]
        result.append(tf.pack(channel_list))
    return tf.pack(result)

def tf_emd(inputs, labels):
    inputs = tf_accumulate(inputs)
    labels = tf_accumulate(labels)

    return tf.reduce_sum(tf.abs(inputs - labels), axis=1)

def tf_var(scores):
    index = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    index_tensor = tf.convert_to_tensor(index)
    mindex_tensor = tf_normlize_to_one_score(scores)
    result = tf.reduce_sum(tf.square(index_tensor - mindex_tensor) * scores, axis=1)
    return result

def tf_crop_rect(img, df, i):
    rec_t = df.rect[i]
    img_t = img[i, rec_t[0]:rec_t[1], rec_t[2]:rec_t[3], :]
    return img_t
    
def tf_photorealism_loss(img, df, i, is_our):
    rec_t = df.rect[i]
    img_t = img[i, rec_t[0]:rec_t[1], rec_t[2]:rec_t[3], :]
    img_t = tf.image.rot90(img_t, 4 - tf.floordiv(df.rot[i], 2))
    img_t = tf.cond(tf.equal(tf.mod(df.rot[i], 2), 0), lambda: img_t, lambda: tf.image.flip_left_right(img_t))
    img_t = tf.transpose(img_t, [1, 0, 2])
    img_r = tf.reshape(img_t, [-1, 3])
    h = rec_t[1] - rec_t[0]
    w = rec_t[3] - rec_t[2]
    k = tf.cast((h - 2) * (w - 2), tf.float32)
    if is_our:
        epsilon1 = 1
        e = tf.constant(np.sqrt(epsilon1), dtype=tf.float32, shape=[1, 3])
        img_r = tf.concat(0, [img_r, e])
        mat_t_r = df.csr_mat_r[i]
        mat_t_g = df.csr_mat_g[i]
        mat_t_b = df.csr_mat_b[i]
        img_r_b, img_r_g, img_r_r = tf.split(1, 3, img_r)
        d_mat_r = tf.sparse_tensor_dense_matmul(mat_t_r, img_r_r)
        d_mat_g = tf.sparse_tensor_dense_matmul(mat_t_g, img_r_g)
        d_mat_b = tf.sparse_tensor_dense_matmul(mat_t_b, img_r_b)
        result_r = tf.reduce_sum(img_r_r * d_mat_r)
        result_g = tf.reduce_sum(img_r_g * d_mat_g)
        result_b = tf.reduce_sum(img_r_b * d_mat_b)
        result = tf.reduce_mean(tf.pack([result_r, result_b, result_g])) / k
    else:
        mat_t = df.csr_mat[i]
        d_mat = tf.sparse_tensor_dense_matmul(mat_t, img_r)
        result = tf.reduce_sum(img_r * d_mat) / (k * 3)
    return result

def tf_imgradient(tensor):
    B, G, R = tf.unpack(tensor, axis=-1)
    tensor = tf.pack([R, G, B], axis=-1)
    tensor = tf.image.rgb_to_grayscale(tensor)
    #tensor = tensor * 255;
    sobel_x = tf.constant([[1, 0, -1], [2, 0, -2], [1, 0, -1]], tf.float32)
    sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
    sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
    #tensor = tf.pad(tensor, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
    fx = tf.nn.conv2d(tensor, sobel_x_filter, strides=[1,1,1,1], padding='VALID')
    fy = tf.nn.conv2d(tensor, sobel_y_filter, strides=[1,1,1,1], padding='VALID')
    g = tf.sqrt(tf.square(fx) + tf.square(fy))
    return g

def matlab_style_gauss2D(shape, sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def tf_imgaussfilt(tensor, sigma):
    fs = int(2 * np.ceil(2 * sigma) + 1)
    kern = tf.constant(matlab_style_gauss2D((fs, fs), sigma), tf.float32)
    kern = tf.reshape(kern, [fs, fs, 1, 1])
    tensor = tf.pad(tensor, [[0, 0], [fs//2, fs//2], [fs//2, fs//2], [0, 0]], 'SYMMETRIC')
    g = tf.nn.conv2d(tensor, kern, strides=[1,1,1,1], padding='VALID')
    return g

def tf_clip_loss(img, ori, df, i):
    rec_t = df.rect[i]
    img_t = img[i, rec_t[0]:rec_t[1], rec_t[2]:rec_t[3], :]
    ori_t = ori[i, rec_t[0]:rec_t[1], rec_t[2]:rec_t[3], :]
    img_t = tf.image.rgb_to_grayscale(img_t)
    ori_t = tf.image.rgb_to_grayscale(ori_t)
    img_o = tf.zeros(shape=tf.shape(img_t), dtype=img_t.dtype)
    img_b = tf.select(img_t <  0, ori_t - img_t, img_o)
    img_b = tf.select(ori_t == 0,         img_o, img_b)
    img_f = img_b
    return tf.reduce_sum(tf.square(img_f))

def tf_improving_loss(score_b, score_a, leak):
    diff = score_a - score_b
    diff_sign = tf.sign(diff)
    diff_abs = diff_sign * diff
    loss = diff_sign * tf.sqrt(diff_abs)
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    loss = f1 * loss + f2 * abs(loss)
    return tf.reduce_mean(loss)

def tf_comparison_loss(guess, label):
    shift = 0.2
    label_zero_index = tf.equal(label, tf.constant(0, dtype=label.dtype))
    label = tf.select(label > 0, label + shift, label - shift)
    weight_t = tf.select(label_zero_index, shift*tf.sign(-guess), label)
    return label * guess

def tf_log10(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def tf_log2(x):
  numerator = tf.log(x)
  denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

def tf_gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    return input_layer + noise

def tf_crop_to_patch(inputs, labels, patch_size, channel, seed):
    inputs_list = tf.unpack(inputs, axis=0)
    labels_list = tf.unpack(labels, axis=0)
    assert(len(inputs_list) == len(labels_list))
    for i in range(len(inputs_list)):
        crop_size = [patch_size, patch_size, channel * 2]
        concat_tensor = tf.concat(concat_dim=2, values=[inputs_list[i], labels_list[i]])
        concat_tensor = tf.random_crop(concat_tensor, crop_size, seed=seed)
        split0, split1 = tf.split(2, 2, concat_tensor)
        inputs_list[i] = split0
        labels_list[i] = split1

    return tf.pack(inputs_list), tf.pack(labels_list)

def tf_random_crop_resize(inputs, labels, img_size, channel, scale, seed):
    assert(scale > 0 and scale <= 1)
    if scale == 1:
        return inputs, labels
    minval = int(round(img_size * scale))
    inputs_list = tf.unpack(inputs, axis=0)
    labels_list = tf.unpack(labels, axis=0)
    assert(len(inputs_list) == len(labels_list))
    for i in range(len(inputs_list)):
        # noise_std = tf.random_uniform([ ], minval=0, maxval=0.01, dtype=tf.float32, seed=seed)
        crop_size = tf.random_uniform([2], minval=minval, maxval=img_size, dtype=tf.int32, seed=seed)
        crop_size = tf.concat(0, [crop_size, [channel * 2]])
        concat_tensor = tf.concat(concat_dim=2, values=[inputs_list[i], labels_list[i]])
        concat_tensor = tf.random_crop(concat_tensor, crop_size, seed=seed)
        concat_tensor = tf.image.resize_images(concat_tensor, [img_size, img_size], method=tf.image.ResizeMethod.BICUBIC)
        split0, split1 = tf.split(2, 2, concat_tensor)
        inputs_list[i] = split0 #gaussian_noise_layer(split0, noise_std)
        labels_list[i] = split1

    return tf.pack(inputs_list), tf.pack(labels_list)

def tf_rgb_to_lab(srgb):
    with tf.name_scope("rgb_to_lab"):
        srgb_pixels = tf.reshape(srgb, [-1, 3])

        with tf.name_scope("srgb_to_xyz"):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("xyz_to_cielab"):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def tf_lab_to_rgb(lab):
    with tf.name_scope("lab_to_rgb"):
        lab_pixels = tf.reshape(lab, [-1, 3])

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope("cielab_to_xyz"):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope("xyz_to_srgb"):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def tf_preprocess_lab(lab):
    with tf.name_scope("preprocess_lab"):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
        return tf.stack([L_chan / 100, (a_chan + 110) / 220, (b_chan + 110) / 220], axis=3)

def tf_deprocess_lab(lab):
    with tf.name_scope("deprocess_lab"):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=3)
        return tf.stack([L_chan * 100, (a_chan * 220) - 110, (b_chan * 220) - 110], axis=3)