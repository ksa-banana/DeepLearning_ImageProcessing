import os
import pandas as pd
import cv2
import numpy as np




def crop_cat_left_ear(dir):
    # image path
    image_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/13371_18106_bundle_archive" + "/" + dir)
    download_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/image/crop_cat_left_ear_img")


    # 파일 리스트 불러오기
    file_list = os.listdir(image_path)
    image_list = [file for file in file_list if file.endswith(".jpg")]


    count = 0

    # 파일 한개씩 resize
    for image_name in image_list:


        # 좌표값 읽기
        points_df = pd.read_csv(image_path + "/" + image_name + ".cat", sep=" ", header=None)


        # 리스트로 바꾸기
        points = points_df.values.tolist()


        # 이미지 읽기
        image = cv2.imread(image_path + "/" + image_name)
        height, width, channel = image.shape
        # cv2.imshow('cat', image)


        # 꼭지점
        i = 13
        x1, x2, x3 = int(points[0][i]), int(points[0][i + 2]), int(points[0][i + 4])
        y1, y2, y3 = int(points[0][i + 1]), int(points[0][i + 3]), int(points[0][i + 5])


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

        # half_len = max - min
        # half_len = int(half_len * (2 / 3))


        # if x1 > x2 | x1 > x3:
        #     half_len =
        # elif x2 >= x3:
        #     half_len = x2 - x1
        # else:
        #     half_len = x3 - x1



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
        img_01_cut = image.copy()
        cut = img_01_cut[y - half_len:y + half_len, x - half_len:x + half_len]



        # 이미지 저장
        try:
            cv2.imwrite(download_path + "/left_ear_" + str(count) + ".jpg", cut)
            print("/left_ear_" + str(count) + ".jpg 출력 완료")

            count = count + 1

        except:
            print(count)








if __name__ == "__main__":
    crop_cat_left_ear("CAT_00")
    # crop_cat_left_ear("CAT_01")
    # crop_cat_left_ear("CAT_02")
    # crop_cat_left_ear("CAT_03")
    # crop_cat_left_ear("CAT_04")
    # crop_cat_left_ear("CAT_05")
    # crop_cat_left_ear("CAT_06")
