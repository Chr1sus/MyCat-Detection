from importlib.resources import path

from PIL import Image
import os
import sys
import cv2
import os,glob

from os import listdir,makedirs

from numpy import save

Directory = "/home/chrisus/proyecto3/data/Test/My_Cat"
SaveDir= "/home/chrisus/proyecto3/data/Test/My_Cat"

for file_name in os.listdir(Directory):
    print("Se esta procesando %s" % file_name)
    output=Image.open(os.path.join(Directory, file_name)).convert('RGB')
    #x,y = image.size
    #new_dimensions = (64,64)
    #output=image.resize(new_dimensions, Image.ANTIALIAS)
    #output = image.transpose(method=Image.ROTATE_90)

    output_file_name= os.path.join(SaveDir, file_name)
    output.save(file_name.replace("png", "jpg") ,quality=95 )

print("Terminado")

'''
path = "/home/chrisus/proyecto3/data/Test/My_Cat"
dstpath= "/home/chrisus/proyecto3/data/Test/My_Cat"
from os.path import isfile,join

files = [f for f in listdir(path) if isfile(join(path,f))] 

for image in files:
    try:
        img = cv2.imread(os.path.join(path,image))
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath,image)
        cv2.imwrite(dstPath,gray)
    except:
        print ("{} is not converted".format(image))

'''


