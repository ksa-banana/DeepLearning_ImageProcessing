import random
import dlib, cv2, os
import pandas as pd
import numpy as np

# 파일 열기 & 순서 shuffle
img_size = 224
dirname = 'TNRCAT_01'
base_path = 'C:/Users/USER/Documents/GitHub/DeepLearning_ImageProcessing/KNR_20200731/TnrCatDataset/%s' % dirname
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list)

# 저장할 dataset 형태
dataset = {
  'imgs': [],
  # 'lmks': [],
  'bbs': []
}


# 이미지 resize
def resize_ear_img(im, landmarks):
    # 꼭지점
    i = 13
    x1, x2, x3 = int(landmarks[6][0]), int(landmarks[7][0]), int(landmarks[8][0])
    y1, y2, y3 = int(landmarks[6][1]), int(landmarks[7][1]), int(landmarks[8][1])

    # 무게중심
    x = int((x1 + x2 + x3) / 3)
    y = int((y1 + y2 + y3) / 3)

    # 왼쪽 귀
    left_ear = np.array([[x1, y1], [x2, y2], [x3, y3]])

    # 필터 가로,세로 크기 구하기
    if y2 >= y1:
        Ymax = y2
        Ymin = y1
    elif y3 >= y2:
        Ymax = y3
        Ymin = y2
    else:
        Ymax = y1
        Ymin = y3

    if x2 >= x1:
        Xmax = x2
        Xmin = x1
    elif x3 >= x2:
        Xmax = x3
        Xmin = x2
    else:
        Xmax = x1
        Xmin = x3

    Yhalf_len = Ymax - Ymin
    Xhalf_len = Xmax - Xmin

    if Yhalf_len >= Xhalf_len:
        half_len = Yhalf_len
        half_len = int(half_len * (2 / 3))
    else:
        half_len = Xhalf_len
        half_len = int(half_len * (2 / 3))


    old_size = im.shape[:2]  # old_size is in (height, width) format


    # 가로 크기 체크(좌표값이 이미지를 벗어나지 않는지)
    if (x + half_len) > old_size[1]:
        half_len = old_size[1] - x
    elif (x - half_len) < 0:
        half_len = x

    # 세로 크기 체크(좌표값이 이미지를 벗어나지 않는지)
    if (y + half_len) > old_size[0]:
        half_len = old_size[0] - y
    elif (y - half_len) < 0:
        half_len = y

    # 이미지 crop
    img_01_cut = im.copy()
    cut = img_01_cut[y - half_len:y + half_len, x - half_len:x + half_len]

    Ltop = [x-half_len, y-half_len]
    Rbottom = [x+half_len, y+half_len]

    return cut, Ltop, Rbottom


for f in file_list:
  if '.cat' not in f:
    continue

  # read landmarks
  pd_frame = pd.read_csv(os.path.join(base_path, f), sep='\t', header=None)
  # pd_frame.as_matrix deprecated -> values 사용
  tmp = pd_frame.values[0][1:]
  landmarks = (pd_frame.values[0][1:]).reshape((-1, 2))

  # load image
  img_filename, ext = os.path.splitext(f)

  img = cv2.imread(os.path.join(base_path, img_filename))


  # resize image and relocate landmarks
  # img, ratio, top, left = resize_img(img)
  img, Ltop, Rbottom = resize_ear_img(img, landmarks)
  # landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
  bb = np.array([Ltop, Rbottom])

  dataset['imgs'].append(img)
  # dataset['lmks'].append(landmarks.flatten())
  dataset['bbs'].append(bb.flatten())

  # for l in landmarks:
  #   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

  # cv2.imshow('img', img)
  # if cv2.waitKey(0) == ord('q'):
  #   break

np.save('C:/Users/USER/Desktop/test/preprocess/%s.npy' % dirname, np.array(dataset))