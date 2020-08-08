from imgaug import augmenters as iaa
import numpy as np
import cv2
import os


# 이미지 읽어오는 함수
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images


# 이미지 저장 함수
def write_images(name, number, images):
    for i in range(0, len(images)):
        # 이미지 저장할 경로 설정을 여기서 한다.
        cv2.imwrite('.\\%s\\%s_%d.jpg' % (name, number, i), images[i])
    print("image saving complete")


# 여러 폴더에 한번에 저장하기
def imagewriterfunction(folder, images):
    for i in range(0, len(images)):
        write_images(folder, str(i), images[i])
    print("all images saved to folder")


# 이미지 증강 코드
def augmentations1(images):
    seq1 = iaa.Sequential([
        iaa.AverageBlur(k=(2, 7)),
        iaa.MedianBlur(k=(3, 11))
    ])

    seq2 = iaa.ChannelShuffle(p=1.0)
    seq3 = iaa.Dropout((0.05, 0.1), per_channel=0.5)
    seq4 = iaa.Sequential([
        iaa.Add((-15, 15)),
        iaa.Multiply((0.3, 1.5))
    ])
    print("image augmentation beginning")
    img1 = seq1.augment_images(images)
    print("sequence 1 completed......")
    img2 = seq2.augment_images(images)
    print("sequence 2 completed......")
    img3 = seq3.augment_images(images)
    print("sequence 3 completed......")
    img4 = seq4.augment_images(images)
    print("sequence 4 completed......")
    print("proceed to next augmentations")
    list = [img1, img2, img3, img4]
    return list


if __name__ == '__main__':
    # 이미지 읽어올 경로
    photos = '.\\Dataset_Augmentation\\'
    folders = os.listdir(photos)

    img_file_list = []
    files = os.listdir(photos)

    # 이미지 이름
    for file in os.walk(photos):
        for i in file[2]:
            img_file_list.append(i[0:4])

    photos1 = load_images_from_folder(os.path.join(photos, folders[0]))
    # photos2 = load_images_from_folder(os.path.join(photos, folders[1]))
    # photos3 = load_images_from_folder(os.path.join(photos, folders[2]))

    photo_augmented1234 = augmentations1(photos1)  # 이미지 증강 0,1,2,3 이 리스트 형태로 있다
    
    # 이미지 이름, 저장
    for i in img_file_list:
        write_images('.\\Result_TNR_CAT_Augmentation\\', i, photo_augmented1234[0])


