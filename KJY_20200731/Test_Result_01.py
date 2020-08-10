import sys, cv2, os
from keras.models import load_model, Sequential
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD


img_size = 224

# model path
# model_path = os.path.join('C:/Users/server/Documents/ProjectTnr/model/')
# model_name = '2020_08_10_09_19_29.h5'
#ears_model = load_model('./Models/ears_1.h5')

# image path
# image_path = os.path.join('C:/Users/server/Documents/ProjectTnr/dataset/test/')
# image_name = 'tnr_cat_ear.jpg'

base_path = './Samples/ears'
file_list = sorted(os.listdir(base_path))

# load model
ears_model = load_model('./Models/ears_1.h5')


def resize_img(im):
    old_size = im.shape[:2]  # old_size is in (height, width) format
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_im, ratio, top, left


for f in file_list:
    if '.jpg' not in f:
        continue

    img = cv2.imread(os.path.join(base_path, f))
    # resize image
    # predict bounding box
    img, ratio, top, left = resize_img(img)
    #input_image = resize_img(img)
    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

    # predict model
    pre_label = ears_model.predict(inputs)

    print(pre_label)


# load model
# model = load_model('./Models/ears_1.h5')

# # load image
# image = cv2.imread(image_path + image_name)

# resize image
# input_image = resize_image(model)
# inputs = (model.astype('float32') / 255).reshape((1, img_size, img_size, 3))
#
# # predict model
# pre_label = model.predict_classes(inputs)
#
# print(pre_label)