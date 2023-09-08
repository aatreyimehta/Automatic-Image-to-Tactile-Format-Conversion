#The purpose of Resize.py is to resize all grayscale/rgb target images to 256x256 dimension and save them with ".tiff" extension as required by the Training model 
#Note: First run Resize.py and then run RemovePng.py on all the tactile1 folders(for horizontal barcharts, vertical barcharts and bezier data).

#importing python packages
from PIL import Image
import math
import os, sys

#folder path
path = "/home/student/Project/.../data/tactile1/"
dirs = os.listdir( path )

#taking input image and resizing it to square shape with 256x256 pixels
def make_square(im, min_size=256, fill_color=(255, 255, 255, 255)):
    x, y = im.size
    size = 256
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

def resize():
    for item in dirs:
        if os.path.isfile(path+item) and "-r" not in path+item:
            im = Image.open(path+item)
            x,y = im.size
            f, e = os.path.splitext(path+item)
            if x == y:   #if image is already square but of greater dimension than 256x256, image is resized to 256x56
                
                imResize = im.resize((256,256), Image.ANTIALIAS)
                #imResize.save(f + '.tiff', 'TIFF', quality=90)
                #imResize = imResize.convert('RGBA')
                #f,e = os.path.splitext(destPath+item)
                print(f+'.tiff')
                imResize.save(f + '.tiff','TIFF')
                print(f)
                print(e)
            else:   #if the image is rectangular, resize to 256x256
            	if x>y:
            	    ratio = 256/x
            	    x = 256
            	    y = math.floor(y*ratio)
            	else:
            	    ratio = 256/y
            	    y = 256
            	    x = math.floor(x*ratio)
            	im = im.resize((x,y), Image.ANTIALIAS)
            	imResize = make_square(im)
            	imResize.save(f + '.tiff','TIFF')   #final image is stored with a ".tiff" extension
                
        
resize()
