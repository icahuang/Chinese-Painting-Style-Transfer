#coding=utf-8

import os, sys

path = './data/cut/all'
dirs = os.listdir(path)
os.chdir(path)
print("dirs:")
print(dirs)


i = 0
for dir in dirs:
	 os.rename(str(dir),str(i) + '.jpg')
	 i += 1