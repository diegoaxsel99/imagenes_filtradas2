# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:52:27 2020

@author: Matador
"""
import skimage as sk
import matplotlib.pyplot as plt
import spacelib as sp
import cv2
import numpy as np
from scipy import ndimage

class folder():
    
    def __init__(self):
        self.mean=[]
        self.gaussian=[]
        self.brightness_mean=[]
        self.contrast_mean=[]
        self.brightness_gaussian=[]
        self.contrast_gaussian=[]

def plotting(folder,title):
    
    plt.figure()
    cont=0
    
    for i in range(3):
        dim=cont+i+3
        
        plt.subplot(1,3,i+1)
        plt.title(title[0]+str(dim)+"X"+str(dim))
        sk.io.imshow(folder.gaussian[i])
        cont=cont+1
    
    plt.figure()
    cont=0
    for i in range(3):
        dim=cont+i+3
        
        plt.subplot(1,3,i+1)
        plt.title(title[1]+str(dim)+"X"+str(dim))
        sk.io.imshow(folder.mean[i])
        cont=cont+1
        
def show(folder,folder2,folder3):
    
    plt.figure()
    
    n=np.array(([1,3,5,7]));
    
    legend=["filtro promedio a pedal"," filtro promedio cv2","filtro promedio scipy"]
    
    plt.subplot(1,2,1)
    plt.title("brillo ")
    
    plt.plot(n,folder.brightness_mean)
    plt.plot(n,folder2.brightness_mean)
    plt.plot(n,folder3.brightness_mean)
    
    plt.legend(legend)
    plt.xlabel("dimension de la mascara")
    plt.ylabel("brillo")
    
    plt.subplot(1,2,2)
    plt.title("brillo ")
    
    plt.plot(n,folder.brightness_gaussian)
    plt.plot(n,folder2.brightness_gaussian)
    plt.plot(n,folder3.brightness_gaussian)
    
    plt.legend(legend)
    plt.xlabel("dimension de la mascara")
    plt.ylabel("brillo")
    
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.title("contraste ")
    
    plt.plot(n,folder.contrast_mean)
    plt.plot(n,folder2.contrast_mean)
    plt.plot(n,folder3.contrast_mean)
    
    plt.legend(legend)
    plt.xlabel("dimension de la mascara")
    plt.ylabel("contraste")
    
    plt.subplot(1,2,2)
    plt.title("contraste ")
    
    plt.plot(n,folder.contrast_gaussian)
    plt.plot(n,folder2.contrast_gaussian)
    plt.plot(n,folder3.contrast_gaussian)
    
    plt.legend(legend)
    plt.xlabel("dimension de la mascara")
    plt.ylabel("contraste")
    
def hist(folder,title):
    
    plt.figure()
    cont=0
    for i in range(3):
        dim=cont+i+3
        
        plt.subplot(1,3,i+1)
        plt.hist(folder.mean[i].ravel(),bins=256)
        plt.title(title+" filtro promedio de mascara "+str(dim)+"x"+str(dim))
        
        cont=cont+1
        
    plt.figure()
    cont=0
    for i in range(3):
        dim=cont+i+3
        
        plt.subplot(1,3,i+1)
        plt.hist(folder.gaussian[i].ravel(),bins=256)
        plt.title(title+" fitro gaussiano de mascara "+str(dim)+"x"+str(dim))
        
        cont=cont+1
    
filename='imagenes/lung2_sp.png'
img=sk.io.imread(filename)

kernel_mean=[]
kernel_gaussian=[]

cont=0
for i in range(3):
    dim=cont+i+3
    kernel_mean.append(sp.kernel_mean(dim,norm=False))
    kernel_gaussian.append(sp.kernel_gaussian(dim,norm=False))
    cont=cont+1

system=folder()
system2=folder()
my=folder()

cont=0
for i in range(3):
    dim=cont+i+3
    
    k1=1/np.sum(kernel_gaussian[i])
    k2=1/np.sum(kernel_mean[i])
    
    system.gaussian.append(cv2.GaussianBlur(img,(dim,dim),0))
    my.gaussian.append(sp.Filter(kernel_gaussian[i],np.copy(img).astype(np.uint8),edge=False,Type='linear'))
    system2.gaussian.append(ndimage.convolve(img,k1*kernel_gaussian[i],mode='reflect'))
    
    system.mean.append(cv2.blur(img,(dim,dim)))
    my.mean.append(sp.Filter(kernel_mean[i],np.copy(img).astype(np.uint8),edge=False,Type='linear'))
    system2.mean.append(ndimage.convolve(img,k2*kernel_mean[i],mode='reflect'))
    
    cont=cont+1

brillo1=sp.brightness(np.copy(img))

my.brightness_mean.append(brillo1)
my.brightness_gaussian.append(brillo1)

system.brightness_mean.append(brillo1)
system.brightness_gaussian.append(brillo1)

system2.brightness_mean.append(brillo1)
system2.brightness_gaussian.append(brillo1)


contraste1=sp.contrast(np.copy(img),brillo1)

my.contrast_mean.append(contraste1)
my.contrast_gaussian.append(contraste1)

system.contrast_mean.append(contraste1)
system.contrast_gaussian.append(contraste1)

system2.contrast_mean.append(contraste1)
system2.contrast_gaussian.append(contraste1)


for i in range(3):
    
    my.brightness_gaussian.append(sp.brightness(np.copy(my.gaussian[i])))
    my.brightness_mean.append(sp.brightness(np.copy(my.mean[i])))
    
    system.brightness_gaussian.append(sp.brightness(np.copy(system.gaussian[i])))
    system.brightness_mean.append(sp.brightness(np.copy(system.mean[i])))
    
    system2.brightness_gaussian.append(sp.brightness(np.copy(system2.gaussian[i])))
    system2.brightness_mean.append(sp.brightness(np.copy(system2.mean[i])))
    
    my.contrast_gaussian.append(sp.contrast(np.copy(my.gaussian[i]),np.copy(my.brightness_gaussian[i+1])))
    my.contrast_mean.append(sp.contrast(np.copy(my.mean[i]),np.copy(my.brightness_mean[i+1])))
    
    
    system.contrast_gaussian.append(sp.contrast(np.copy(system.gaussian[i]),np.copy(system.brightness_gaussian[i+1])))
    system.contrast_mean.append(sp.contrast(np.copy(system.mean[i]),np.copy(system.brightness_mean[i+1])))
    
    system2.contrast_gaussian.append(sp.contrast(np.copy(system.gaussian[i]),np.copy(system2.brightness_gaussian[i+1])))
    system2.contrast_mean.append(sp.contrast(np.copy(system.mean[i]),np.copy(system2.brightness_mean[i+1])))
    

plotting(system,['gaussiana cv2 ','promedio cv2 '])
plotting(my,['gaussiana a pedal ','promedio a pedal '])
plotting(system2,['gaussiana ndimage ','promedio ndimage '])

show(my,system,system2)
hist(my,"pedal")
hist(system,"cv2")
hist(system2,"ndimage")


