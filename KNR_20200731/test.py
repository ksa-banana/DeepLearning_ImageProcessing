from keras.models import Model, load_model
import cv2
import os

img_size = 224

# model path
model_path = os.path.join('C:/Users/server/Documents/ProjectTnr/model/')
model_name = '2020_08_10_09_19_29.h5'

# image path
image_path = os.path.join('C:/Users/server/Documents/ProjectTnr/dataset/test/')
image_name = 'tnr_cat_ear.jpg'

def resize_image(image):

  old_size = image.shape[:2]
  ratio = float(img_size) / max(old_size)
  new_size = tuple([int(x*ratio) for x in old_size])

  resize_image = cv2.resize(image, (new_size[1], new_size[0]))
  delta_w = img_size - new_size[1]
  delta_h = img_size - new_size[0]

  top, bottom = delta_h // 2, delta_h - (delta_h // 2)
  left, right = delta_w // 2, delta_w - (delta_w // 2)

  new_image = cv2.copyMakeBorder(resize_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])

  return new_image



# load model
model = load_model(model_path + model_name)

# load image
image = cv2.imread(image_path + image_name)

# resize image
input_image = resize_image(image)
inputs = (image.astype('float32') / 255).reshape((1, img_size, img_size, 3))

# predict model
pre_label = model.predict_classes(inputs)

print(pre_label)
