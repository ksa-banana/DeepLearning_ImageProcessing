import random
import dlib
import cv2
import os
import pandas as pd
import numpy as np

# 이미지를 resize할 크기
img_size = 224
dirname = 'CAT_06'
base_path ='./Cat_Images/%s' % dirname
file_list = sorted(os.listdir(base_path))
random.shuffle(file_list)

# 최종적으로 저장할 데이터셋
dataset={
    'imgs':[],
    'lmks': [],
    'bbs': []
}

# 이미지를 원하는 크기로 자르고 정사각형으로 만드는 함수
def resize_img(im):
    # old_size is in (height, width) format
    old_size = im.shape[:2]
    ratio = float(img_size) / max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = img_size - new_size[1]
    delta_h = img_size - new_size[0]
    top, bottom = delta_h//2, delta_h - (delta_h//2)
    left, right = delta_w//2, delta_w - (delta_w//2)
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return new_im, ratio, top, left


for f in file_list:
    if '.cat' not in f:
        continue

    # read landmarks
    # pd.read_csv(): 구분자로 분리 구성된 파일을 읽음
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    # 9*2 형태로 만들어주고 int 타입으로 변환
    landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2))
    # bounding box
    bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)]).astype(np.int)
    center = np.mean(bb, axis=0)

    # 고양이 얼굴 크기
    face_size = max(np.abs(np.max(landmarks, axis=0) - np.min(landmarks, axis=0)))
    # 새로운 bounding box 지정 -> 타이트하게 자르지 않고 넓게 자른다.
    new_bb = np.array([center-face_size*0.6, center+face_size*0.6]).astype(np.int)
    # 마이너스가 되지 않도록 한다.
    # np.clip(): 행렬안의 값을 min, max로 제한
    new_bb = np.clip(new_bb, 0, 99999)
    # 자른 이미지의 새로운 랜드마크 생성
    new_landmarks = landmarks - new_bb[0]

    # load image
    # 이미지 불러오기
    img_filename, ext = os.path.splitext(f)
    img = cv2.imread(os.path.join(base_path, img_filename))
    # 이미지 자르기
    new_img = img[new_bb[0][1]:new_bb[1][1], new_bb[0][0]:new_bb[1][0]]


    # resize image and relocate landmarks
    img, ratio, top, left = resize_img(new_img)
    # resize한 만큼 landmark 재조정
    new_landmarks = ((new_landmarks * ratio) + np.array([left, top])).astype(np.int)

    # 데이터셋에 저장
    dataset['imgs'].append(img)
    dataset['lmks'].append(new_landmarks.flatten())
    dataset['bbs'].append(new_bb.flatten())

# for l in new_landmarks:
#   cv2.circle(img, center=tuple(l), radius=1, color=(255, 255, 255), thickness=2)

# cv2.imshow('img', img)
# if cv2.waitKey(0) == ord('q'):
#   sys.exit(1)

np.save('dataset/lmks_%s.npy' % dirname, np.array(dataset))