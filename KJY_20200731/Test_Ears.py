import sys, cv2, os
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

from Helper import resize, test

img_size = 224
base_path = './Samples'
file_list = sorted(os.listdir(base_path))

print('model load start')

# 저장한 모델을 넣어주는 부분

bbs_model = load_model('./Models/bbs_1.h5')
lmks_model = load_model('./Models/lmks_1.h5')

print('model load finish')
print('testing start')

# testing
for f in file_list:
    if '.jpg' not in f:
        continue

    print('imread')

    img = cv2.imread(os.path.join(base_path, f))
    ori_img = img.copy()
    result_img = img.copy()

    print('predict bounding box')

    # predict bounding box
    img, ratio, top, left = resize.resize_img(img)

    inputs = (img.astype('float32') / 255).reshape((1, img_size, img_size, 3))
    pred_bb = bbs_model.predict(inputs)[0].reshape((-1, 2))

    # compute bounding box of original image
    ori_bb = ((pred_bb - np.array([left, top])) / ratio).astype(np.int)

    # compute lazy bounding box for detecting landmarks
    center = np.mean(ori_bb, axis=0)
    face_size = max(np.abs(ori_bb[0] - ori_bb[1]))
    new_bb = np.array([
        center - face_size * 0.6,
        center + face_size * 0.6
    ]).astype(np.int)
    new_bb = np.clip(new_bb, 0, 99999)

    print('predict landmarks')

    # predict landmarks
    face_img = ori_img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]
    face_img, face_ratio, face_top, face_left = resize.resize_img(face_img)

    face_inputs = (face_img.astype('float32') / 255).reshape((1, img_size, img_size, 3))

    pred_lmks = lmks_model.predict(face_inputs)[0].reshape((-1, 2))

    # compute landmark of original image
    new_lmks = ((pred_lmks - np.array([face_left, face_top])) / face_ratio).astype(np.int)
    ori_lmks = new_lmks + new_bb[0]

    # visualize
    cv2.rectangle(ori_img, pt1=tuple(ori_bb[0]), pt2=tuple(ori_bb[1]), color=(255, 255, 255), thickness=2)

    # 좌표 저장
    pointList = []
    for i, l in enumerate(ori_lmks):
        cv2.putText(ori_img, str(i), tuple(l), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(ori_img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)
        print(l)
        pointList.append(l[0])
        pointList.append(l[1])

    print(pointList)

    #파일 쓰기
    filename, ext = os.path.splitext(f)
    file = open('D://workspace_KSA//workspace_PyCharm//Deeplearning_20200730//Result//Point//'+filename+'.jpg'+'.cat', 'w')
    file.write('9')
    for i in pointList:
        data = '\t{}'.format(i)
        file.write(data)
    file.close()

print('testing finish')