import numpy as np
import os
import glob
from PIL import Image
import shutil
import random
import glob
import cv2

def mirar_size(s,path1,path2,):
  l = os.listdir(path)


def blurrear(im):
 return cv2.GaussianBlur(im,(181,181),0)
 
def hacerdir(path):
  try:
    os.mkdir(path)
  except FileExistsError:
    pass
    

def extract_from_combine(path1,path2):
  l = os.listdir(path1)
  for i in l:
    im = Image.open(path1+i)
    im = np.array(im)
    imr = im[:,1023:,:]
    imr = Image.fromarray(imr)
    imr.save(path2+i,'png')
  
def divide_image(image,name):
  im = np.array(image)
  c = 0
  for i in range(0,im.shape[0],int(im.shape[0]/4)):
    for j in range(0,im.shape[1],int(im.shape[0]/4)):
      c += 1
      im_aux = im[i:i+int(im.shape[0]/4),j:j+int(im.shape[0]/4)]
      im_aux = Image.fromarray(im_aux)
      
      im_aux.save(PATH9+name[:-4]+'_'+str(c)+'.png','PNG')


def train_val_test(list_images,path,trainpath,valpath,testpath,ntrain=2400,nval=200,ntest=376,max=11000,format = '.png'):
    a =4
    if (ntrain+ntest+nval)>max:
        ntrain=5000
        nval=1000
        ntest = 1000
    for i,image in enumerate(list_images):
        if i <ntrain:
            shutil.copy(path + image, trainpath+image[:-a]+format)
        elif i<(ntrain+nval):
            shutil.copy(path+image,valpath+image[:-a]+format)
        elif i< (ntrain+nval+ntest):
            shutil.copy(path+ image,testpath+image[:-a]+format)
            
def quitar_ultima_banda(image):
    im = np.array(image)
    im = im[:,:,0:3]
    im = Image.fromarray(im)
    
    return im
    
def resize_images(path1,path2,insize = 512,outsize=32):
  l = os.listdir(path1)
  for i in l:
    im = Image.open(path1+i)
    quitar_ultima_banda(im)
    im = im.resize((outsize,outsize))
    im = im.resize((insize,insize))
    im.save(path2+i[:-4]+'.png','png')
    

def carn_dataset(path):
  l = os.listdir(path)
  os.mkdir(path+'train/')
  os.mkdir(path+'test/')
  l1 = os.listdir(path+l[0])
  random.shuffle(l1)
  for i in l:
    
    os.mkdir(path+'train/'+i)
    os.mkdir(path+'test/'+i)
    
    
    for num,j in enumerate(l1):
      if num<(len(l1)*0.8):
        shutil.move(path+i+'/'+j,path+'train/'+i+'/'+j)
      else:
        shutil.move(path+i+'/'+j,path+'test/'+i+'/'+j)
      

parser = argparse.ArgumentParser()
parser.add_argument("--tipoDataset", type=str,default='25cmDecimado')
parser.add_argument("--pathICGC", type=str)
parser.add_argument("--pathSentinel", type=str,default=' ')
args = parser.parse_args()

path1 = 'groundtruth_'+args.tipoDataset
path2 = 'entrada_'+args.tipoDataset
hacerdir(path1)
hacerdir(path2)
if args.tipoDataset == '25cmDecimado':
  resize_images(path1,path2,,insize = 1024,outsize=32)
elif args.tipoDataset == '50cmDecimado':
  resize_images(path1,path2,,insize = 512,outsize=32)
elif args.tipoDataset == '1mDecimado':
  resize_images(path1,path2,,insize = 256,outsize=32)
elif args.tipoDataset == '25cmDistorsionado':
  for i in os.listdir(path1):
    im = Image.open(path1+i)
    im = blurrear(im)
    im.save(path2+i[:-4]+'.png','png')
elif args.tipoDataset == '25cmSentinel':
  for i in os.listdir(args.pathSentinel):
    im = Image.open(args.pathSentinel+i)
    im.save(path2+i[:-4]+'.png','png')

l = os.listdir(path1)
random.shuffle(l)
hacerdir(path1+'train/')
hacerdir(path1+'test/')
hacerdir(path1+'val/')
hacerdir(path2+'train/')
hacerdir(path2+'test/')
hacerdir(path2+'val/')
train_val_test(l,path1,path1+'train/',path1+'val/',path1+'test/')
train_val_test(l,path2,path2+'train/',path2+'val/',path2+'test/')
    





















