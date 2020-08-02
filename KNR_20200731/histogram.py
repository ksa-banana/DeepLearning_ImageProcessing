import matplotlib.pyplot as plt
import os
import cv2


# image path
image_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/image/crop_cat_left_ear_img")
download_path = os.path.join("C:/Users/knr64/Documents/tnr_cat_project/image/crop_cat_left_ear_img")


# 파일 리스트 불러오기
file_list = os.listdir(image_path)
image_list = [file for file in file_list if file.endswith(".jpg")]


#
count_list = []
size_list = []
max = 0
min = 1000000





# 이미지 사이즈 min, max 구하기
for image_name in image_list:

    # 이미지 읽기
    image = cv2.imread(image_path + "/" + image_name)
    height, width, channel = image.shape

    # 가장 큰 사이즈 저장
    if height > max:
        max = height

    # 가장 작은 사이즈 저장
    if height < min:
        min = height



print("min:" + str(min) + ", max: "+ str(max))

# 연속된 숫자 리스트
# x축 초기화
size_list = list(range(min, max+1))
# print(size_list)


# 이미지 사이즈 리스트
# y축 초기화
count_list = [0 for _ in range(max-min+1)]


# 사이즈 개수 세기
for image_name in image_list:

    # 이미지 읽기
    image = cv2.imread(image_path + "/" + image_name)
    height, width, channel = image.shape

    # count 추가
    count_list[height-min-1] = count_list[height-min-1] + 1



# print(count_list)


# 그래프 그리기
plt.plot(size_list, count_list)

plt.xlabel('이미지 크기')
plt.ylabel('개수')
plt.title('crop image')

plt.show()
