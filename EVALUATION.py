import tensorflow as tf
import numpy as np
from FUNCTION import *

def psnr(input, label, shave):
    assert (label.dtype == input.dtype)
    if shave != 0:
        input = input[shave:-shave, shave:-shave, :]
        label = label[shave:-shave, shave:-shave, :]
    diff = np.int32(label) - np.int32(input)
    mse = np.mean(diff ** 2.)
    if mse == 0:
        return 1e6
    max_i = np.iinfo(label.dtype).max
    return 10*np.log10(max_i * max_i / mse)

def tf_psnr(input, label, shave):
    assert (label.dtype == input.dtype)
    if shave != 0:
        input = input[shave:-shave, shave:-shave, :]
        label = label[shave:-shave, shave:-shave, :]
    diff = tf.cast(label, tf.int32) - tf.cast(input, tf.int32)
    mse = tf.reduce_mean(tf.cast(diff, tf.float32) ** 2.)
    return tf.cond(tf.equal(mse, 0), lambda: tf.constant(1e6, dtype=mse.dtype), lambda: 20*np.log10(label.dtype.max)-10*tf_log10(mse))

def tf_psnr_float(input, label, shave, max_i, single_channel, is_circle, scale):
    assert (label.dtype == input.dtype)

    if shave != 0:
        if single_channel:
            if shave != 0 and len(input.get_shape()) == 2:
                input = input[shave:-shave, shave:-shave]
                label = label[shave:-shave, shave:-shave]
            elif shave != 0 and len(input.get_shape()) == 3:
                input = input[:, shave:-shave, shave:-shave]
                label = label[:, shave:-shave, shave:-shave]
            else:
                assert False, 'DIMENSION ERROR'
        else:
            if len(input.get_shape()) == 2:
                input = input[shave:-shave, shave:-shave]
                label = label[shave:-shave, shave:-shave]
            elif len(input.get_shape()) == 3:
                input = input[shave:-shave, shave:-shave, :]
                label = label[shave:-shave, shave:-shave, :]
            elif len(input.get_shape()) == 4:
                input = input[:, shave:-shave, shave:-shave, :]
                label = label[:, shave:-shave, shave:-shave, :]
            else:
                assert False, 'DIMENSION ERROR'

    diff = tf.sub(label, input)
    if is_circle:
        h = diff
        abs_dh = tf.abs(h)
        middle = 0.5 * max_i
        mask_1 = tf.cast( abs_dh > middle, tf.float32)
        mask_2 = tf.cast((mask_1 * h) < 0, tf.float32)
        mask_3 = tf.cast((mask_1 * h) > 0, tf.float32)
        diff = (h + (middle * mask_2) - (middle * mask_3)) * 2

    mse = tf.reduce_mean(tf.square(diff)) * scale + 1e-10
    return 20*np.log10(max_i)-10*tf_log10(mse)

def ssim(img1, img2, shave, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    assert(img1.dtype == img2.dtype)
    if shave != 0:
        img1 = img1[shave:-shave, shave:-shave, :]
        img2 = img2[shave:-shave, shave:-shave, :]
    size = (size - 1) // 2
    window = fspecial_gauss(size, sigma)
    L = np.iinfo(img1.dtype).max
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    mu1 = ndimage.convolve(img1, window)
    mu2 = ndimage.convolve(img2, window)
    mu1 = mu1[size:-size, size:-size, :]
    mu2 = mu2[size:-size, size:-size, :]
    mu1_sq  = mu1*mu1
    mu2_sq  = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = ndimage.convolve(img1*img1, window)
    sigma2_sq = ndimage.convolve(img2*img2, window)
    sigma12   = ndimage.convolve(img1*img2, window)
    sigma1_sq = sigma1_sq[size:-size, size:-size, :] - mu1_sq
    sigma2_sq = sigma2_sq[size:-size, size:-size, :] - mu2_sq
    sigma12   = sigma12  [size:-size, size:-size, :] - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = np.mean(value)
    return value

def tf_fspecial_gauss(size, sigma):
    x_data, y_data = np.mgrid[-size:size + 1, -size:size + 1]

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x*x + y*y)/(2.0*sigma*sigma)))
    g = tf.expand_dims(g, -1)
    g = tf.expand_dims(g, -1)
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, shave, L, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    assert(img1.dtype == img2.dtype)
    img1 = tf.transpose(img1, [2, 0, 1])
    img2 = tf.transpose(img2, [2, 0, 1])
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    if shave != 0:
        img1 = img1[:, shave:-shave, shave:-shave, :]
        img2 = img2[:, shave:-shave, shave:-shave, :]
    window = tf_fspecial_gauss((size-1)//2, sigma)
    K1 = 0.01
    K2 = 0.03
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    img1 = tf.cast(img1, tf.float32)
    img2 = tf.cast(img2, tf.float32)
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq  = mu1*mu1
    mu2_sq  = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1], padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1], padding='VALID') - mu2_sq
    sigma12   = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1], padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value =  ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value
