#coding=utf-8

import os

def rename_files(path):
    file_list = os.listdir(path)
    print("Original files' names:")
    print(file_list)
    #cwd = os.getcwd()
    #print(cwd)
    os.chdir(path)
    
    #(2)for each files, rename filename
    i = 0
    for file_name in file_list:
        os.rename(file_name, str(i)+'.jpg')
        i = i + 1
    print("Current files' names:")
    print(file_list)


path = '/Users/icahuang/Desktop/source_files/StyleTransfer/Data_augmentation/data'
rename_files(path)
