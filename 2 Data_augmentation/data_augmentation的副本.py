#coding=utf-8

import tensorflow as tf
import numpy as np
import os
import skimage
import glob
from PIL import Image
import imgaug as ia
from imgaug import augmenters as iaa
import scipy
import cv2


ia.seed(1)

IMAGE_SIZE = 2500

imgCount = 106
height = 2500
width = 2500
dims = 3

input_path = './data/original'
output_path = './data/augmented'


'''
input: image Diractory path

output: 
1. imgs = images array(type: numpy.ndarray, shape = (imgCount, height, width, dims))
2. file_name_list = 输入图像的文件名列表
'''
def create_image(image_path):
    # type: (object) -> object
    imgs = np.zeros((imgCount, height, width, dims))
    file_name_list = os.listdir(image_path)
    i = 0

    for imgFile in file_name_list:
        #print(imageFile)
        # 打开图像并转化为数字矩阵(128x128x3)
        img = np.array(Image.open(os.path.join(image_path, imgFile)))
        imgs[i] = img
        # print(imgs[i])
        i += 1

    #print(imgs.shape)
    return imgs, file_name_list


#The data type of the element of imgArr must be numpy.Array
def save_image(imgArr, output_path):
    #检查存放输出图像的文件夹是否已经被创建，若没有则先创建
    if os.path.isdir(output_path):
        pass
    else:
        os.mkdir(output_path)

    i = 0
    for imgFile in imgArr:
        #scipy.misc.imsave(output_path + '/' +  '%s_%d.jpg' %(name, i), imgFile)
        path = os.path.join(output_path, '%s_%d.jpg' % ('augmented', i))
        scipy.misc.imsave(path, imgFile)
        i = i + 1

    return



def data_augmentation_tf(img_dir_path):
    img_name_list = os.listdir(img_dir_path)
    img_path_list = []
    count = len(img_name_list)

    for i in range(count):
        img_path_list.append(os.path.join(img_dir_path, img_name_list[i]))
    print("img_path_list:")
    print(img_path_list)

    #augmented_img_list = 作为增强后的数据的容器
    img = np.array(Image.open(img_path_list[0]))
    tup1 = []
    tup2 = []
    for j in range(len(img.shape)):
        tup2.append(img.shape[j])
    shape = tup1 + tup2
    print(shape)
    augmented_img_list = np.zeros(shape)
    print(augmented_img_list.shape)


    for i in range(count):
        with tf.Session() as sess:
            raw_img = tf.gfile.Open(img_path_list[i], 'rb').read()  #bytes, 把文件存储到string中返回
            img_data = tf.image.decode_image(raw_img)    #convert the input bytes string into a “Tensor of type dtype.
            #img_data = tf.image.convert_image_dtype(img_data, tf.float32)
            img_data = tf.image.convert_image_dtype(img_data, tf.uint8)
            # cv2.imshow('raw_image', img_data.eval())     # 转为ndarray类型
            print('raw_shape:', img_data.eval().shape)

            """resize"""
            # img_data = tf.image.resize_images(img_data.eval(), (224, 224))

            """crop and black pad"""
            # img_data = tf.image.resize_image_with_crop_or_pad(img_data, target_height=1000,
            #                                                   target_width=1000)

            """按照倍数中心裁剪, 倍数=(0, 1]"""
            # img_data = tf.image.central_crop(img_data, central_fraction=0.2)

            """pad """
            # img_data = tf.image.pad_to_bounding_box(img_data, offset_height=10, offset_width=10,
            #                                         target_height=576+10, target_width=576+10)

            """crop"""
            # img_data = tf.image.crop_to_bounding_box(img_data, 40, 40, 576-40, 576-40)

            """extract
                                        o----->
                                        |
                                        |
                                        v
                                        x
            """
            # img_data = tf.image.extract_glimpse(tf.expand_dims(img_data, 0), size=[100, 100],
            #                                     offsets=tf.reshape(tf.constant([-.4, .4], dtype=tf.float32), [1, 2]))

            """roi pooling 必要操作 boxes为长宽比值!!! """
            # img_data = tf.image.crop_and_resize(tf.expand_dims(img_data, 0), boxes=[[0/576, 0/576, 1, 1]],
            #                                     box_ind=[0], crop_size=[100, 100])

            """上下翻转/左右/转置翻转/90度旋转---(random_)flip_up_down/flip_left_right/transpose/rot90"""
            img_data = tf.image.rot90(img_data, k=1)

            """Converting Between Colorspaces"""
            """灰度"""
            # img_data = tf.image.rgb_to_grayscale(img_data)
            """图像亮度[-1, 1]"""
            # img_data = tf.image.adjust_brightness(img_data, delta=-.7)
            """随机图像亮度"""
    #         img_data = tf.image.random_brightness(img_data, max_delta=0.6)
            """随机对比度"""
    #         img_data = tf.image.random_contrast(img_data, lower=0, upper=4)
            """随机色调"""
            # img_data = tf.image.random_hue(img_data, 0.5)
            """随机饱和度"""
    #         img_data = tf.image.random_saturation(img_data, lower=0, upper=2)
            """图片标准化    (x - mean) / max(stddev, 1.0/sqrt(image.NumElements()))"""
            # img_data = tf.image.per_image_standardization(img_data)
            """draw boxes"""
            # img_data = tf.image.draw_bounding_boxes(tf.expand_dims(img_data, 0), [[[0.1, 0.2, 0.5, 0.9]]])
            print('Tensor:', img_data)
            print(img_data.eval().shape)
            print(img_data.eval())
            
            # #plt显示图像
            # plt.figure(1)
            # plt.imshow(img_data.eval())
            # plt.show()

            # #cv2显示图像
            # cv2.imshow('changed', img_data.eval())
            # cv2.waitKey()

            # np.append(augmented_img_list, img_data.eval())
            augmented_img_list[i] = img_data.eval()

    print("augmented_img_list:")
    print(augmented_img_list)
    print("augmented_img_list.shape:")
    print(augmented_img_list.shape)
    return augmented_img_list



def data_augmentation():    
    seq = iaa.Sequential([
        #iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(1), # horizontally flip 100% of the images
        #iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
    ])



def main():
    #导入所有图像到dir列表中,file_name_list保存的是所有输入图像的文件名（plus: 不包含路径）
    #dir, file_name_list = create_image(input_path)

    aug_img_list = data_augmentation_tf(input_path)

    #将所有图像都保存到output_path路径下
    save_image(aug_img_list, output_path)



if __name__ == '__main__':
  main()




