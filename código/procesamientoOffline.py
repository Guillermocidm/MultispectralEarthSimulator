import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.restoration import estimate_sigma
from PIL import Image
import cv2
import torch
from torchvision.transforms import ToTensor,Normalize


def differenceByPixel(im1,im2):
	"""
	metrica para medir la diferencia por píxel de dos imagenes
	"""
    return np.sum(np.abs(im1 - im2)) / np.prod(im2.shape)


def blurProcess(path1,path2):
	"""
	Recibe dos ficheros de imagenes 
	genera una lista de nombres y otra con su respectivo valor de 
	diferencia por píxel entre las dos imagenes
	"""
    images_list = os.listdir(path_images)
    names = np.array([])
    metrics = np.array([])
    for i,img_path in enumerate(images_list):
        names = np.append(names,img_path)

        #   IMAGENES DE AVION SOLO BLUR MASCARA (MASK_VALUE) 201
        img_icgc = cv2.imread(PATH_ICGC+img_path)
        img_icgc = cv2.cvtColor(img_icgc, cv2.COLOR_BGR2RGB)
        img_icgc = cv2.resize(img_icgc, (32, 32), interpolation=cv2.INTER_AREA)
        img_icgc = cv2.resize(img_icgc, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        blur = cv2.GaussianBlur(img_icgc, (MASK_VALUE,MASK_VALUE),0)

        #   IMAGENES SAT RESIZE A 1024
        img_sat = cv2.imread(PATH_SAT + img_path)
        img_sat = cv2.cvtColor(img_sat, cv2.COLOR_BGR2RGB)
        img_sat = cv2.resize(img_sat,(1024,1024),interpolation=cv2.INTER_LINEAR)

        #   NORMALIZAR DATOS EN CADA BANDA (RGB)
        blur_norm = (blur - np.mean(blur,(0,1))) / np.std(blur,(0,1))
        img_sat_norm = (img_sat - np.mean(img_sat,(0,1))) / np.std(img_sat,(0,1))

        #   APLICAR METRICA
        metrics = np.append(metrics,metric(blur_norm,img_sat_norm))
        print(metric(blur_norm,img_sat_norm))

    #    DEVUELVO ARRAYS DE NOMBRES Y METRICAS ORDENADO
    names = names[metrics.argsort()]
    metrics.sort()
    return names, metrics

def estimateNoise(images_path):
	"""
	Recibe un conjunto de imagenes en un fichero
	genera una lista de nombres y otra con su respectivo valor de 
	ruido
	
	"""
    icgc_list = os.listdir(images_path)
    names = np.array([])
    noise = np.array([])
    for i, img_path in enumerate(icgc_list):
        names = np.append(names,img_path)
        img = cv2.imread(PATH_50cm+img_path)
        noise = np.append(noise,estimate_sigma(img, multichannel=True, average_sigmas=True))
        

    names = names[noise.argsort()]
    noise.sort()
    return names,noise

def extractMeanStd(path):
	"""
	Recibe un un conjunto de imagenes de un fichero 
	devuelve un diccionario con la media y desviación de cada imagen,
	la media y desviación del conjunto de las imagenes 
	"""
    cnt = 0
    fst_moment = torch.zeros(3)
    snd_moment = torch.zeros(3)
    path_list = os.listdir(path)
    dict={}
    for i, data in enumerate(path_list):
        print(i, data)

        img = Image.open(path+ data)
        img_tensor = ToTensor()(img)
        b, h, w = img_tensor.shape
        nb_pixels = h * w
        sum_ = torch.sum(img_tensor, dim=[1, 2])
        sum_of_square = torch.sum(img_tensor ** 2, dim=[1, 2])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        fst_moment_2 = (0 * fst_moment + sum_) / (nb_pixels)
        snd_moment_2 = torch.std(img_tensor,dim=[1,2])
        dict[data] = (fst_moment_2.tolist(),snd_moment_2.tolist())
        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2),dict