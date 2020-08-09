import random
import dlib, cv2, os
import pandas as pd
import numpy as np

img_size = 224

dataset = {
  'imgs' : [],
  'lbs' : []
}


def crop_image(image_name):

  # 좌표값 읽기
  points_df = pd.read_csv(image_path + "/" + image_name + ".cat", sep=" ", header=None)
  points = (points_df.values[0][1:-1]).reshape((-1, 2))


  # 왼쪽 귀 lmks
  left_ear = points[6:]
  x_list = []
  y_list = []
  for row in left_ear:
    x_list.append(int(row[0]))
    y_list.append(int(row[1]))


  # 왼쪽 귀 bounding box
  bbox = []
  bbox.append([min(x_list) - 20, min(y_list) - 20])
  bbox.append([max(x_list) + 20, max(y_list) + 20])



  # 이미지 읽기
  image = cv2.imread(image_path + "/" + image_name)
  height, width, channel = image.shape


  # 크기 체크(좌표값이 이미지를 벗어나지 않는지)
  if bbox[0][0] < 0 : bbox[0][0] = 0
  if bbox[0][1] < 0 : bbox[0][1] = 1
  if bbox[1][0] > width : bbox[1][0] = width
  if bbox[1][1] > height : bbox[1][1] = height


  # 이미지 crop
  copy_image = image.copy()
  crop_image = copy_image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
  height, width, channel = crop_image.shape


  # 이미지 출력
  # cv2.imshow('crop', crop_image)
  # cv2.waitKey(0)
  # cv2.destroyAllWindows()

  return crop_image, width, height




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




if __name__ == "__main__":

  count = 0

  dataset = {
    'imgs': [],
    'lbs': []
  }

  dir_name = 'CAT_05'

  # directory name
  dir_name = dir_name
  tnr_dir_name = 'TNR_'+ dir_name

  # image path
  image_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/13371_18106_bundle_archive/" + dir_name)
  tnr_image_path = os.path.join(
    'C:/Users/knr64/Downloads/DeepLearning_ImageProcessing-master/Dataset_TNR_CAT/CAT_Augmentation/' + tnr_dir_name)
  download_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/dataset")

  # 파일 리스트 불러오기
  file_list = os.listdir(image_path)
  image_list = [file for file in file_list if file.endswith(".jpg")]
  tnr_file_list = os.listdir(tnr_image_path)
  tnr_image_list = [file for file in tnr_file_list if file.endswith(".jpg")]

  # CAT
  for image_name in image_list:

    # crop image
    ear_image, width, height = crop_image(image_name)

    # resize image
    new_image = resize_image(ear_image)

    # dataset에 image와 label 입력
    dataset['imgs'].append(new_image)
    dataset['lbs'].append(0)

    print(str(count) + ' 완료')
    count = count + 1


  # TNR CAT
  for image_name in tnr_image_list:

    # 이미지 읽기
    tnr_ear_image = cv2.imread(tnr_image_path + "/" + image_name)

    # resize image
    new_image = resize_image(tnr_ear_image)

    # dataset에 image와 label 입력
    dataset['imgs'].append(new_image)
    dataset['lbs'].append(1)

    print(str(count) + ' 완료')
    count = count + 1


  # combine & shuffle
  combine = list(zip(dataset['imgs'], dataset['lbs']))
  random.shuffle(combine)
  dataset['imgs'], dataset['lbs'] = zip(*combine)


  # save image
  np.save(download_path + '/%s.npy' % dir_name, np.array(dataset))





