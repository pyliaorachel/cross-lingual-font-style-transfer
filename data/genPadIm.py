import cv2
import os
import numpy as np 
path_to_dir = 'pingfang/'
path_out = 'pingfang/tmpout/'
all_image = [im for im in os.listdir(path_to_dir) if im.endswith('.png')]

#print(all_image)
for I in all_image:
    img = cv2.imread(path_to_dir + I, flags=cv2.IMREAD_COLOR)
    height, width, _ = img.shape
    paddingNum = abs(width - height) # try catch assert
    n1 = n2 = (int)(paddingNum/2)
    if paddingNum % 2 != 0:
        n1 += 1
    if width >= height:
        img = cv2.copyMakeBorder(img,n1,n2,0,0,cv2.BORDER_CONSTANT,value=(255,255,255)) # top, bottom, left, right
    else:
        img = cv2.copyMakeBorder(img,0,0,n1,n2,cv2.BORDER_CONSTANT,value=(255,255,255)) # top, bottom, left, right
    #height, width, _ = img_padding.shape
    img = cv2.resize(img,(128,128))
    cv2.imwrite( path_out + I, img)


