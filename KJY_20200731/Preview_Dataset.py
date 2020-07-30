import random
import dlib
import cv2
import os
import pandas as pd
import numpy as np

# 데이터 파일 리스트
dirname = 'CAT_00'
base_path ='./Cat_Images/%s' % dirname
file_list = sorted(os.listdir(base_path))

for f in file_list:
    if '.cat' not in f:
        continue

    # read landmarks
    # pd.read_csv(): 구분자로 분리 구성된 파일을 읽음
    pd_frame = pd.read_csv(os.path.join(base_path, f), sep=' ', header=None)
    # 9*2 형태로 만들어주고 int 타입으로 변환
    landmarks = (pd_frame.to_numpy()[0][1:-1]).reshape((-1, 2)).astype(np.int)

    # load image
    img_filename, ext = os.path.splitext(f)
    # cv.imread(): 이미지를 읽음
    img = cv2.imread(os.path.join(base_path, img_filename))

    # visualize
    for l in landmarks:
        # cv2.circle(): 이미지에 원을 그림
        cv2.circle(img, center=tuple(l), radius=1, color=(0, 0, 255), thickness=2)
        
        # cv2.imshow(): 이미지를 윈도우에 뛰움
        cv2.imshow('img', img)
        if cv2.waitKey(0) == ord('q'):
            break
