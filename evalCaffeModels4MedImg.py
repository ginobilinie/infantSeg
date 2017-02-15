    
'''
Target: evaluate your trained caffe model with the medical images. I use simpleITK to read medical images (hdr, nii, nii.gz, mha and so on)  
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
from imgUtils import dice
import os
import h5py
import numpy as np  
import scipy.io as scio

# Make sure that caffe is on the python path:
caffe_root = '/usr/local/caffe3/'  # this is the path in GPU server
import sys
sys.path.insert(0, caffe_root + 'python')
print caffe_root + 'python'
import caffe

caffe.set_device(1) #very important
caffe.set_mode_gpu()
### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt') #for training
protopath='/your/path/to/prototxtfiles/'
mynet = caffe.Net(protopath+'pelvic_deploy_3d.prototxt',protopath+'pelvic_fcn_3d_binary1_iter_276000.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(mynet.blobs.keys(), mynet.params.keys()))

d1=32
d2=32
d3=32
d=[d1,d2,d3]
step1=1
step2=1
step3=1
step=[step1,step2,step3]
    
def cropCubic(matFA,matSeg,fileID,d,step,rate):
    eps=1e-5
    [row,col,leng]=matFA.shape
    cubicCnt=0
    matOut=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))
    used=np.zeros((matSeg.shape[0],matSeg.shape[1],matSeg.shape[2]))+eps
    #fid=open('trainxxx_list.txt','a');
    for i in range(0,row-d[0],step[0]):
        for j in range(0,col-d[1],step[1]):
            for k in range(0,leng-d[2],step[2]):
                volSeg=matSeg[i:i+d[0],j:j+d[1],k:k+d[2]]
                #if np.sum(volSeg)<eps
                #    continue
                #cubicCnt=cubicCnt+1
                volFA=matFA[i:i+d[0],j:j+d[1],k:k+d[2]]
                mynet.blobs['dataMR'].data[0,0,...]=volFA
                mynet.forward()
                temppremat = mynet.blobs['softmax'].data[0].argmax(axis=0) #Note you have add softmax layer in deploy prototxt
                matOut[i:i+d[0],j:j+d[1],k:k+d[2]]=matOut[i:i+d[0],j:j+d[1],k:k+d[2]]+temppremat;
                used[i:i+d[0],j:j+d[1],k:k+d[2]]=used[i:i+d[0],j:j+d[1],k:k+d[2]]+1;
    matOut=matOut/used
    return matOut

#this function is used to compute the dice ratio
def dice(im1, im2,tid):
    im1=im1==tid #make it boolean
    im2=im2==tid #make it boolean
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    dsc=2. * intersection.sum() / (im1.sum() + im2.sum())
    return dsc

def main():
    path='/your/path/to/data/'
    
    #ids=[1,2,3,4,5,6,7,8,9,10,11]
    ids=[1,2,3,4,6,7,8,10,11,12,13]
    for i in range(0,len(ids)):
		myid=ids[i]    
		datafilename='img%d.mhd'%myid #provide a sample name of your filename of data here
		datafn=os.path.join(path,datafilename)
		labelfilename='img%d_seg.mhd'%myid  # provide a sample name of your filename of ground truth here
		labelfn=os.path.join(path,labelfilename)
		imgOrg=sitk.ReadImage(datafn)
		mrimg=sitk.GetArrayFromImage(imgOrg)
		mu=np.mean(mrimg)
		maxV=np.max(mrimg)
		minV=np.min(mrimg)
		mrimg=mrimg
		mrimg=(mrimg-mu)/(maxV-minV)
    	#img=sitk.GetImageFromArray(mrimg)
    	#img.SetSpacing(imgtmp.GetSpacing())
    	#img.SetOrigin(imgtmp.GetOrigin())
    	#img.SetDirection(imgtmp.GetDirection())
    	#normfilter=sitk.NormalizeImageFilter()
    	#img=normfilter.Execute(img)
		labelOrg=sitk.ReadImage(labelfn)
		labelimg=sitk.GetArrayFromImage(labelOrg)
		labelimg=labelimg/10
    	#you can do what you want here for for your label img
		fileID='%d'%myid
		rate=1
		matOut=cropCubic(mrimg,labelimg,fileID,d,step,rate)
    	# here you can make it round to nearest integer 
    	#now we can compute dice ratio
    
if __name__ == '__main__':     
    main()
