#coding=utf-8

import os
import cv2

#设置输出图像的大小 此处默认为正方形图像 即大小 = output_size * output_size
output_size = 2500
input_path = "./data/千里江山图.jpg"
output_path = "./data/cut/"
start_px = 500

#start_px: 从左往右数从第几个px开始裁剪
def image_cut(input_path, output_path, output_size, start_px=0):
	img = cv2.imread(input_path)  #shape=(2500, 55000, 3)
	height, width, channel = img.shape
	output_path = output_path + 'spx%d_size%d' %(start_px, output_size)
	#图像导出到path文件夹里
	if os.path.isdir(output_path):
		pass
	else:
		os.mkdir(output_path)

	#计算裁剪后总共会输出多少个图像
	dataset_size = int((width - start_px)/output_size)
	#裁剪并输出保存
	for i in range(0, dataset_size):
	    cropped = img[0:output_size, start_px + i*output_size:start_px + (i+1)*output_size]
	    path = output_path + '/spx%d_cut_image_%d.jpg' %(start_px, i)
	    #print(path)
	    print(cropped.shape)
	    cv2.imwrite(path, cropped)


image_cut(input_path, output_path, output_size, start_px)