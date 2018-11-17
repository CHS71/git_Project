#이미지 data 변환에 필요한 코드




# -*- coding: utf-8 -*-

import os
import numpy as np
import cv2

IMAGE_SIZE = 64

#1
def resize_with_pad(image, height=IMAGE_SIZE, width=IMAGE_SIZE):

    def get_padding_size(image):
        h, w, _ = image.shape
        longest_edge = max(h, w)
        top, bottom, left, right = (0, 0, 0, 0)
        if h < longest_edge:
            dh = longest_edge - h
            top = dh // 2
            bottom = dh - top
        elif w < longest_edge:
            dw = longest_edge - w
            left = dw // 2
            right = dw - left
        else:
            pass
        return top, bottom, left, right

    top, bottom, left, right = get_padding_size(image)
    BLACK = [0, 0, 0]
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value=BLACK)

    resized_image = cv2.resize(constant, (height, width))

    return resized_image


##1
images = []
labels = []
def traverse_dir(path):
    for file_or_dir in os.listdir(path):                             #입력한 경로의 파일과 폴더 목록 리스트를 loop문 돌림
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))  #절대경로얻기
        # print(abs_path)
        if os.path.isdir(abs_path):  # dir   os.path.isdir("있니없니_검사할폴더경로")  TRUE or FALSE
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpeg') or file_or_dir.endswith('.png'):   # 만약 내가 찾고 싶은 문자열의 형태가 마지막에 있는지 확인
                image = read_image(abs_path)
                images.append(image)
                labels.append(path)

    return images, labels





#
def read_image(file_path):
    image = cv2.imread(file_path)
    image = resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image

#2
def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)
    labels = np.array([0 if label.endswith('oracleU') else 1 for label in labels])  #보스사진을 모아둔 폴더 이름의 마지막을 boss로 해야할것

    return images, labels


###########
# print(traverse_dir('C:\\video'))


# print(traverse_dir('C:\\video'))
# print(extract_data('C:\\video'))


# print(extract_data('C:\\video'))
