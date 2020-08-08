# 이미지 자르기

import os
import matplotlib.pyplot as plt
import pandas as pd


def extract_file(path):
    img_file_list = []
    cat_file_list = []
    files = os.listdir(path)

    for file in files:

        if file[-3:] == 'jpg':

            img_file_list.append(file)

        elif file[-3:] == 'cat':

            cat_file_list.append(file)

    return img_file_list, cat_file_list


def preprocessing_cat_file(folder, cat_file):
    x_locs = []
    y_locs = []
    locs = []
    loc_list = []
    feature_loc = pd.read_csv('D:/workspace_KSA/workspace_PyCharm/Deeplearning_20200730/Dataset_TNR_CAT/' + '/' + folder + '/' + cat_file, header=None)
    feature_loc = (feature_loc[0][0]).split('\t')
    #feature_loc.pop()
    feature_loc = list(map(int, feature_loc))

    for i in range(7, (feature_loc[0] * 2) + 1):

        if (i % 2 != 0):
            x_locs.append(feature_loc[i])

        else:
            y_locs.append(feature_loc[i])

    for i in range(len(x_locs)):
        locs.append((x_locs[i], y_locs[i]))
        left_ear = locs[:3]
        right_ear = locs[3:]

    le_max_left = min(left_ear)[0]
    le_max_right = max(left_ear)[0]
    le_max_top = min(left_ear, key=lambda x: x[1])[1]
    le_max_bottom = max(left_ear, key=lambda x: x[1])[1]

    re_max_left = min(right_ear)[0]
    re_max_right = max(right_ear)[0]
    re_max_top = min(right_ear, key=lambda x: x[1])[1]
    re_max_bottom = max(right_ear, key=lambda x: x[1])[1]

    loc_list.append(le_max_left)
    loc_list.append(le_max_right)
    loc_list.append(le_max_top)
    loc_list.append(le_max_bottom)
    loc_list.append(re_max_left)
    loc_list.append(re_max_right)
    loc_list.append(re_max_top)
    loc_list.append(re_max_bottom)

    return loc_list


def ear_img_save(cat_image, file_name, loc_list):
    for i in range(2):

        if i == 0:
            left_ear_image = plt.imshow(
                cat_image[(loc_list[2] - 15): (loc_list[3] + 15), (loc_list[0] - 15): (loc_list[1] + 15)])
            plt.axis('off')
            plt.savefig('D:/workspace_KSA/workspace_PyCharm/Deeplearning_20200730/Result_TNR_CAT/left_ear/' + file_name+'.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()
            print(file_name, '왼쪽 귀 저장 완료')

        if i == 1:
            left_ear_image = plt.imshow(
                cat_image[(loc_list[6] - 15): (loc_list[7] + 15), (loc_list[4] - 15): (loc_list[5] + 15)])
            plt.axis('off')
            plt.savefig('D:/workspace_KSA/workspace_PyCharm/Deeplearning_20200730/Result_TNR_CAT/right_ear/' + file_name+'.jpg', bbox_inches='tight', pad_inches=0)
            plt.close()
            return print(file_name, '오른쪽 귀 저장 완료')


if __name__ == '__main__':
    path = 'D:/workspace_KSA/workspace_PyCharm/Deeplearning_20200730/Dataset_TNR_CAT'
    folders = os.listdir(path)

    for folder in folders:

        img_file_list, cat_file_list = extract_file(path + '/' + folder)

        for i in range(len(img_file_list)):

            cat_image = plt.imread(
                'D:/workspace_KSA/workspace_PyCharm/Deeplearning_20200730/Dataset_TNR_CAT' + '/' + folder + '/' +
                img_file_list[i])
            loc_list = preprocessing_cat_file(folder, cat_file_list[i])

            try:
                ear_img_save(cat_image, img_file_list[i][:-4], loc_list)

            except Exception as ex:
                print(ex)
