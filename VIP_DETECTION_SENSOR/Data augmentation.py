#Data augmentation (modified)


import os
from keras.preprocessing.image import ImageDataGenerator,  img_to_array, load_img

from keras.preprocessing.image import random_rotation

def generate_plot_pics_rotation(datagen ,orig_img ):
    dir_augmented_data = 'C:\\video\\rotate'


    ## convert the original image to array
    x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    x = x.reshape((1,) + x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
    i = 0
    Nplot = 8
    for batch in datagen.flow(x ,batch_size=1,
                              save_to_dir=dir_augmented_data,
                              save_prefix="pic",
                              save_format='jpeg'):
        i += 1
        if i > Nplot - 1: ## generate 8 pictures
            break




def generate_plot_pics_horizontal(datagen ,orig_img ):
    dir_augmented_data = 'C:\\video\\horizontal'

    ## convert the original image to array
    x = img_to_array(orig_img)
    ## reshape (Sampke, Nrow, Ncol, 3) 3 = R, G or B
    x = x.reshape((1,) + x.shape)
    ## -------------------------- ##
    ## randomly generate pictures
    ## -------------------------- ##
    i = 0
    Nplot = 8
    for batch in datagen.flow(x ,batch_size=1,
                              save_to_dir=dir_augmented_data,
                              save_prefix="picc",
                              save_format='jpeg'):
        i += 1
        if i > Nplot - 1: ## generate 8 pictures
            break







import PIL, os
from PIL import Image

#os.chdir('/root/imagedatagenerator') # change to directory where image is located
path = 'C:\\video\\the_other'





for file_or_dir in os.listdir(path):  # 입력한 경로의 파일과 폴더 목록 리스트를 loop문 돌림
    abs_path = os.path.abspath(os.path.join(path, file_or_dir))
    print('$',abs_path)
    orig_img= Image.open(str(abs_path))
    # orig_img = picture.rotate(270)




    # %matplotlib inline
    import matplotlib.pyplot as plt

    datagen_r = ImageDataGenerator(rotation_range=30)
    print('@')
    datagen_h = ImageDataGenerator(horizontal_flip=True)
    print('@@')

    generate_plot_pics_rotation(datagen_r,orig_img)
    print('@@@')
    generate_plot_pics_horizontal(datagen_h,orig_img)
    print('@@@@')