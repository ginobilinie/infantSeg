    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  

d1=32
d2=32
d3=32
d=[d1,d2,d3]
step1=10
step2=10
step3=8
step=[step1,step2,step3]
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def cropCubic(matFA,matSeg,fileID,d,step,rate):
    eps=1e-5
    [row,col,len]=matFA.shape
    cubicCnt=0
    estNum=20000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
        
       
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,len-dSeg[2]+1,step[2]):
                volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volSeg)<eps:
                    continue
                cubicCnt=cubicCnt+1
                volFA=matFA[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                trainFA[:,:,:,0,cubicCnt]=volFA
                trainSeg[:,:,:,0,cubicCnt]=volSeg
    #trainFA=float(trainFA)
    #trainSeg=float(trainSeg)
    with h5py.File('./train_%s.h5'%fileID,'w') as f:
        f['dataMR']=trainFA
        f['dataSeg']=trainSeg
    with open('./trainxxx_list.txt','a') as f:
        f.write('./trainxxx.h5\n')
    return cubicCnt
    	
def main():
    path='/home/dongnie/warehouse/xxx/'
    saveto='/home/dongnie/warehouse/prostate'
   
    datafilename='prostate_%dto1_MRI.nii'%id #provide a sample name of your filename of data here
    datafn=os.path.join(path,datafilename)
    labelfilename='prostate_%dto1_CT.nii'%id  # provide a sample name of your filename of ground truth here
    labelfn=os.path.join(path,labelfilename)
    imgOrg=sitk.ReadImage(datafn)
    mrimg=sitk.GetArrayFromImage(imgOrg)
  	mu=np.mean(mrimg)
  	maxV=np.max(mrimg)
  	minV=np.min(mrimg)
  	mrimg=float(mrimg)
  	mrimg=(mrimg-mu)/(maxV-minV)
    #img=sitk.GetImageFromArray(mrimg)
    #img.SetSpacing(imgtmp.GetSpacing())
    #img.SetOrigin(imgtmp.GetOrigin())
    #img.SetDirection(imgtmp.GetDirection())
    #normfilter=sitk.NormalizeImageFilter()
    #img=normfilter.Execute(img)
    labelOrg=sitk.ReadImage(labelfn)
    labelimg=sitk.ReadImage(labelOrg) 
    #you can do what you want here for for your label img
    
    fileID='%d'%id
    rate=1
    cubicCnt=cropCubic(mrimg,labelimg,fileID,d,step,rate)
    
if __name__ == '__main__':     
    main()
