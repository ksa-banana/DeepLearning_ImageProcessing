import matplotlib.pyplot as plt
import os
import pandas as pd
import cv2
from PIL import Image
import numpy as np


#image path
image_path = os.path.join("C:/Users/USER/Downloads/13371_18106_bundle_archive/CAT_00")
down_path = os.path.join("C:/Users/USER/Desktop/test/image")
image_name = "/00000364_000.jpg"


# 이미지 출력
# img_01 = plt.imread(image_path + image_name)
# plt.imshow(img_01)
# plt.show()



#좌표값 읽기
points_df = pd.read_csv(image_path + image_name + ".cat", sep=" ", header=None)
#display(pd.DataFrame(points_df))



#리스트로 바꾸기
points = points_df.values.tolist()


#opencv
#이미지 읽기
image_01 = cv2.imread(image_path + image_name)
height, width, channel = image_01.shape
cv2.imshow('cat', image_01)



# 이미지의 가로, 세로 구하기
height, width, channel = image_01.shape



#꼭지점
i = 13
x1, x2, x3 = int(points[0][i]), int(points[0][i+2]), int(points[0][i+4])
y1, y2, y3 = int(points[0][i+1]), int(points[0][i+3]), int(points[0][i+5])


#무게중심
x = int((x1 + x2 + x3) / 3)
y = int((y1 + y2 + y3) / 3)


#왼쪽 귀
left_ear = np.array([[x1, y1], [x2, y2], [x3, y3]])


#화면에 보여주기
img_01 = cv2.polylines(image_01, [left_ear], True, (255,0,0), 2)
img_01 = cv2.line(image_01, (int(x), int(y)),(int(x), int(y)), (255, 255, 255), 2)
img_01 = cv2.line(image_01, (int(x1), int(y1)),(int(x1), int(y1)), (255, 255, 255), 2)
img_01 = cv2.line(image_01, (int(x2), int(y2)),(int(x2), int(y2)), (255, 255, 255), 2)
cv2.imshow('cat', img_01)

cv2.waitKey(0)
cv2.destroyAllWindows()


# 필터 가로,세로 크기 구하기

max, min = 0, 1024

# 최대값
if x1 >= max:
    max = y1
if x2 >= max:
    max = y2
if x3 >= max:
    max = y3

# 최소값
if x1 <= min:
    min = y1
if x2 <= min:
    max = y2
if x3 <= min:
    max = y3

half_len = max - min


# 가로 크기 체크(좌표값이 이미지를 벗어나지 않는지)
if (x + half_len) > width:
    half_len = width - x
elif (x - half_len) < 0:
    half_len = x

# 세로 크기 체크(좌표값이 이미지를 벗어나지 않는지)
if (y + half_len) > height:
    half_len = height - y
elif (y - half_len) < 0:
    half_len = y


# 이미지 crop
img_01_cut = image_01.copy()
cut = img_01_cut[y-half_len:y+half_len, x-half_len:x+half_len]
# img_01_cut[0:half_len*2, 0:half_len*2] = cut
cv2.imshow('a', cut)

cv2.waitKey(0)
cv2.destroyAllWindows()







# 이미지 저장
cv2.imwrite('cut.jpg', cut)

