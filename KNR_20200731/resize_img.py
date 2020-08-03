import os
import cv2


def resize_image(size):

    # image path
    image_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/image/crop_cat_left_ear_img")
    download_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/image/crop_cat_left_ear_resize_img")

    # 파일 리스트 불러오기
    file_list = os.listdir(image_path)
    image_list = [file for file in file_list if file.endswith(".jpg")]

    count = 0

    # resize
    for image_name in image_list:

        # 이미지 읽기
        image = cv2.imread(image_path + "/" + image_name)

        # resize
        rs_image = cv2.resize(image,dsize=(size, size), interpolation=cv2.INTER_AREA)
        print(image_name)

        # 이미지 저장
        try:
            cv2.imwrite(download_path + "/left_ear_" + str(count) + ".jpg", rs_image)
            print("/left_ear_" + str(count) + ".jpg 출력 완료")

            count = count + 1

        except:
            print(count)




if __name__ == "__main__":
    # 가로 크기 67로 resize
    resize_image(67)
