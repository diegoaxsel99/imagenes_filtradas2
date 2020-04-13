# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:35:05 2020

@author: Matador
"""

import numpy as np

def kernel_mean(dim,norm):
    
    mask=(np.ones((dim,dim)).astype(np.uint8))
    
    if(norm==True):
        return mask*1/np.sum(mask)
    else:
        return mask


def pascal_pyramide(dim):
    
    pascal=[]
    
    pascal.append([1,1])
    vector=[]
    
    for i in range(dim-1):
        vector=[]
        for j in range(pascal[i].__len__()+1):
            
            if(j==0 or j==pascal[i].__len__()):
                vector.append(1)
            else:
                vector.append(pascal[i][j]+pascal[i][j-1])
        
        pascal.append(vector)
        
    return pascal[dim-1]
            
def kernel_gaussian(dim,norm):
    
    vector=pascal_pyramide(dim-1)
    
    mask2=np.ones((dim,dim),np.float64)
    for i in range(dim):
        for j in range(dim):
            
            if(i==0 or i==dim-1):
                mask2[i][j]=vector[j]
            if(j==0 or j==dim-1):
                mask2[i][j]=vector[i]
                
    for i in range(1,dim-1):
        for j in range (1,dim-1):
            mask2[i][j]=mask2[i][0]*mask2[0][j]
   
    if(norm==True):
        return mask2*1/np.sum(mask2)
    else:
        return mask2

            

def mirror(posx,posy,img,Sum,k,x,y):
    
    flagx=-1
    flagy=-1
    
    if(posx>=img.shape[1]):
        posx=img.shape[1]-(posx+1)
        flagx=1
    if(posy>=img.shape[0]):
        posy=img.shape[0]-(posy+1)
        flagy=1
    
    Sum=Sum+k*img[y+flagy*posy][x+flagx*posx]
   
    return Sum
    
def look_around(mask,x,y,img):
    
    dim=len(mask)
    posx=x-int(dim/2)
    posy=y-int(dim/2)
    contx=0
    conty=0
    cont=0
    Sum=0
    
    while(contx!=dim and conty!=dim):
        
        k=mask[int(cont/dim)][int(cont%dim)] 
        cont=cont+1
        
        if(posx<=-1 or posy<=-1 or posx>img.shape[1]-1 or posy>img.shape[0]-1):
            Sum=mirror(posx,posy,img,Sum,k,x,y)
            
            posx=posx+1
            contx=contx+1
            
            if(contx==dim):
                
                posx=x-int(dim/2)
                posy=posy+1
                conty=conty+1
                contx=0
                
        else:
            
            Sum=Sum+k*img[posy][posx]
                
            posx=posx+1
            contx=contx+1
                
            if(contx==dim):
            
                posx=x-int(dim/2)
                posy=posy+1
                contx=0
                conty=conty+1
        
    Sum=Sum/np.sum(mask)
    
    
    return int(Sum)
    
        
def Filter(mask,img,edge,Type):
    
    filtered=img
    
    if(Type=='linear'):
        
        if(edge==True):
        
            for i in range(int(len(mask)/2),img.shape[0]-1):
                for j in range(int(len(mask)/2),img.shape[1]-1):
                
                    filtered[i][j]=look_around(mask,j,i,img)
        else:
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    filtered[i][j]=look_around(mask,j,i,img)
                
    return filtered

def brightness(img):
    
    dim=img.shape
    brillo=0
    
    k=1/(dim[0]*dim[1])
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            brillo=brillo+img[i][j]
    
    brillo=brillo*k
    
    return brillo

def contrast(img,B):
    
    dim=img.shape
    contraste=0
    
    k=1/(dim[0]*dim[1])
    
    for i in range(dim[0]):
        for j in range(dim[1]):
            contraste=contraste+np.power(img[i][j]-B,2)
            
    contraste=np.sqrt(k*contraste)
    
    return contraste
    



                                             




