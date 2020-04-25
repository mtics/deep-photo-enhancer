from DATA import *
from PIL import Image
from numpy import array


def get_normalize_size_shape_method(img):
    [height, width, channels] = img.shape
    if height >= width:
        longerSize = height
        shorterSize = width
    else:
        longerSize = width
        shorterSize = height

    scale = float(FLAGS['data_image_size']) / float(longerSize)
    outputHeight = int(round(height * scale))
    outputWidth = int(round(width * scale))
    return outputHeight, outputWidth


def cpu_normalize_image(img):
    outputHeight, outputWidth = get_normalize_size_shape_method(img)
    outputImg = Image.fromarray(img)
    outputImg = outputImg.resize((outputWidth, outputHeight), Image.ANTIALIAS)
    outputImg = array(outputImg)
    return outputImg
