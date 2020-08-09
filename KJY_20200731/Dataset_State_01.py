# 이미지 파일 개수를 가져오는 모듈
import os
from PIL import Image
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

# 전체 파일 이름 리스트
total_filename = []
# 전체 이미지 리스트
total_images = []
# 폴더명 리스트
filenum = ['00'];

# 이미지 사이즈에 따라서 개수 저장하는 딕셔너리
dic_file = {}

# 정렬된 이미지의 사이즈별 개수
sdict_file = []

# 정렬된 사이즈 리스트
dic_file_keys = []
# 정렬된 사이즈별 이미지 개수
dic_file_values = []

# 해당 디렉토리의 개수 세기
class GetAllFiles:
    def __init__(self):
        pass
    
    # 파일명 가져오는 함수
    def getFiles(self, dir):
        x = 0
        filename = []
        for pack in os.walk(dir):
            print("current directory = %s , sub directory = %s , files = %s " % (pack[0], pack[1], pack[2]))
            for i in pack[2]:
                filename.append(i)
                #print ( ("Dir: %s , there are %s of files") % (dir, str(x)))
                x += 1
        return filename

    # jpg 파일만 저장하는 함수
    def addImage(self, total_filename):
        z = 0
        for l in range(0, len(total_filename)):
            images = []
            for n in total_filename[l]:
                if (n.endswith('.jpg')):
                    images.append(n)
            total_images.insert(z, images)
            z += 1

        sum = 0
        for o in range(0, len(total_images)):
            sum = sum + len(total_images[o])
        print('이미지 총 개수: ', sum)

        return total_images
    
    # 이미지 사이즈 개수에 따라 저장
    def getImageSize(self, total_images):
        for p in range(0, len(total_images)):
            dir = '.\\Dataset_TNR_CAT\\TNR_CAT_0{}'.format(p)
            for q in total_images[p]:
                img = Image.open(dir + '\\{}'.format(q))
                width, height = img.size
                size = str(width) + ',' + str(height)

                if size in dic_file:
                    dic_file[size] += 1
                else:
                    dic_file[size] = 1
        print('이미지 사이즈 종류 개수: ', len(dic_file))
        return dic_file
    
    # 사이즈별 가장 많은 이미지순으로 정렬(내림차순)하고 keys와 values로 저장
    def getSort(self, dic_file):
        sdict_file = sorted(dic_file.items(), reverse=True, key=lambda k: k[1])
        for i in range(0, 11):
            dic_file_keys.append(sdict_file[i][0])
            dic_file_values.append(sdict_file[i][1])
        return dic_file_keys, dic_file_values

    # 표 만들기
    def getTable(self, dic_file_keys, dic_file_values):
        frame = DataFrame({'이미지 개수': dic_file_values, '이미지 크기': dic_file_keys})
        print(frame)

    # Bar 그래프 그리기
    def getGraph(self, dic_file_keys, dic_file_values):
        index = np.arange(len(dic_file_keys))
        plt.rcParams["figure.figsize"] = (20, 5)
        plt.rcParams['lines.linewidth'] = 30
        plt.rcParams['lines.color'] = 'r'
        plt.rcParams['axes.grid'] = True

        plt.bar(index, dic_file_values, width=0.5, bottom=2, align='edge', label='A', color='r', edgecolor='black', linewidth=1.2)
        plt.title('Image size', fontsize=20)
        plt.xlabel('pixel (weight, height)', fontsize=18)
        plt.ylabel('number of images', fontsize=18)
        plt.xticks(index, dic_file_keys, fontsize=10)
        plt.show()


if __name__ == '__main__':
    
    #클래스 선언 및 생성
    f = GetAllFiles()
    y = 0
    # 폴더별로 파일이름을 리스트에 저장
    for k in filenum:
        total_filename.insert(y, f.getFiles((".\\Dataset_TNR_CAT\\TNR_CAT_{}").format(k)))
        y += 1

    f.getSort(f.getImageSize(f.addImage(total_filename)))
    f.getTable(dic_file_keys, dic_file_values)
    f.getGraph(dic_file_keys, dic_file_values)



