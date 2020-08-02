import random
import dlib
import cv2
import os
import pandas as pd
import numpy as np

# 이미지를 resize할 크기
num = ['00', '01', '02', '03', '04', '05', '06']
img_size = 224

for i in num:
    dirname = 'CAT_%s' % i
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
        if '.cat' not in f:
            continue

        # read landmarks
        # pd.read_csv(): 구분자로 분리 구성된 파일을 읽음
        pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
        # 9*2 형태로 만들어주고 int 타입으로 변환
        landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2))

        # load image
        img_filename, ext = os.path.splitext(f)
        # cv.imread(): 이미지를 읽음
        img = cv2.imread(os.path.join(base_path, img_filename))

        # resize image and relocate landmarks
        img, ratio, top, left = resize_img(img)
        # 변화된 랜드마크 주소를 재계산
        landmarks = ((landmarks * ratio) + np.array([left, top])).astype(np.int)
        # bounding box(bb): 고양이 얼굴에 박스로 영역 생성
        # 최소점: 상단 왼쪽(x1, y1) , 최대점: 하단 오른쪽(x2, y2)
        bb = np.array([np.min(landmarks, axis=0), np.max(landmarks, axis=0)])

        dataset['imgs'].append(img)
        dataset['lmks'].append(landmarks.flatten())
        dataset['bbs'].append(bb.flatten())

    # 전처리가 끝난 데이터셋을 CAT_##.npy 이름으로 저장
    np.save('./Dataset/%s.npy' % dirname, np.array(dataset))