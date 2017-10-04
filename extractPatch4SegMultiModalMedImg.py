    
'''
Target: Crop patches for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
for single-scale patches
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np

eps=1e-5
tol=0.95
d1=32
d2=32
d3=32
dFA=[d1,d2,d3] # size of patches of input data
dMR=[d1,d2,d3] # size of patches of input data
dSeg=[32,32,32] # size of pathes of label data
step1=10
step2=8
step3=6
step=[step1,step2,step3]
saveto='/home/dongnie/warehouse/iSeg/iSeg-2017-Training/'
    
'''
Actually, we donot need it any more,  this is useful to generate hdf5 database
'''
def extractPatches4OneSubject(matFA,matMR,matSeg,fileID,d,step,rate):
  
    rate1=1.0/2
    rate2=1.0/4
    [row,col,leng]=matFA.shape
    cubicCnt=0
    estNum=20000
    trainFA=np.zeros([estNum,1, dFA[0],dFA[1],dFA[2]],np.float16)
    trainMR=np.zeros([estNum,1, dMR[0],dMR[1],dMR[2]],np.float16)
    trainSeg=np.zeros([estNum,1,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
#     trainFA2D=np.zeros([estNum, dFA[0],dFA[1],dFA[2]],np.float16)
#     trainSeg2D=np.zeros([estNum,dSeg[0],dSeg[1],dSeg[2]],dtype=np.int8)
    #print 'trainFA shape, ',trainFA.shape
    #to padding for input
    margin1=(dFA[0]-dSeg[0])/2
    margin2=(dFA[1]-dSeg[1])/2
    margin3=(dFA[2]-dSeg[2])/2
    cubicCnt=0
    marginD=[margin1,margin2,margin3]
    #print 'matFA shape is ',matFA.shape
    matFAOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matMROut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    matSegOut=np.zeros([row+2*marginD[0],col+2*marginD[1],leng+2*marginD[2]])
    
    print 'matFAOut shape is ',matFAOut.shape
    matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA
    matMROut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matMR
    matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg
    #for mageFA, enlarge it by padding
    if margin1!=0:
        matFAOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matFAOut[row+marginD[0]:matFAOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matFA[matFA.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matFAOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matFA[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matFAOut[marginD[0]:row+marginD[0],col+marginD[1]:matFAOut.shape[1],marginD[2]:leng+marginD[2]]=matFA[:,matFA.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matFA[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matFAOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matFAOut.shape[2]]=matFA[:,:,matFA.shape[2]-1:leng-marginD[2]-1:-1]
    #for matseg, enlarge it by padding
    if margin1!=0:
        matSegOut[0:marginD[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[marginD[0]-1::-1,:,:] #reverse 0:marginD[0]
        matSegOut[row+marginD[0]:matSegOut.shape[0],marginD[1]:col+marginD[1],marginD[2]:leng+marginD[2]]=matSeg[matSeg.shape[0]-1:row-marginD[0]-1:-1,:,:] #we'd better flip it along the 1st dimension
    if margin2!=0:
        matSegOut[marginD[0]:row+marginD[0],0:marginD[1],marginD[2]:leng+marginD[2]]=matSeg[:,marginD[1]-1::-1,:] #we'd flip it along the 2nd dimension
        matSegOut[marginD[0]:row+marginD[0],col+marginD[1]:matSegOut.shape[1],marginD[2]:leng+marginD[2]]=matSeg[:,matSeg.shape[1]-1:col-marginD[1]-1:-1,:] #we'd flip it along the 2nd dimension
    if margin3!=0:
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],0:marginD[2]]=matSeg[:,:,marginD[2]-1::-1] #we'd better flip it along the 3rd dimension
        matSegOut[marginD[0]:row+marginD[0],marginD[1]:col+marginD[1],marginD[2]+leng:matSegOut.shape[2]]=matSeg[:,:,matSeg.shape[2]-1:leng-marginD[2]-1:-1]
        
    dsfactor = rate
    #actually, we can specify a bounding box along the 2nd and 3rd dimension, so we can make it easier 
    for i in range(0,row-dSeg[0]+1,step[0]):
        for j in range(0,col-dSeg[1]+1,step[1]):
            for k in range(0,leng-dSeg[2]+1,step[2]):
                volSeg=matSeg[i:i+dSeg[0],j:j+dSeg[1],k:k+dSeg[2]]
                if np.sum(volSeg)<eps:
                    continue
                #bgRate=1-np.count_nonzero(matSeg)/np.prod(matSeg.shape)
                #print bgRate
                #if bgRate>tol:
                #    continue;
                
                cubicCnt=cubicCnt+1
                #index at scale 1
            
                
                volFA=matFAOut[i:i+dFA[0],j:j+dFA[1],k:k+dFA[2]]
                volMR=matMROut[i:i+dMR[0],j:j+dMR[1],k:k+dMR[2]]
                trainFA[cubicCnt,0,:,:,:]=volFA #32*32*32
                trainMR[cubicCnt,0,:,:,:]=volMR #32*32*32

                trainSeg[cubicCnt,0,:,:,:]=volSeg#24*24*24

#                 trainFA2D[cubicCnt,:,:,:]=volFA #32*32*32
#                 trainSeg2D[cubicCnt,:,:,:]=volSeg#24*24*24


    trainFA=trainFA[0:cubicCnt,:,:,:,:]
    trainMR=trainMR[0:cubicCnt,:,:,:,:]
    trainSeg=trainSeg[0:cubicCnt,:,:,:,:]

#     trainFA2D=trainFA2D[0:cubicCnt,:,:,:]
#     trainSeg2D=trainSeg2D[0:cubicCnt,:,:,:]

    with h5py.File('./train32range_%s.h5'%fileID,'w') as f:
        f['dataT1']=trainFA
        f['dataT2']=trainMR
        f['dataSeg']=trainSeg
#         f['dataMR2D']=trainFA2D
#         f['dataSeg2D']=trainSeg2D
     
    with open('./train_iSeg_32range_list.txt','a') as f:
        f.write(saveto+'train32range_%s.h5\n'%fileID)
    return cubicCnt

#to remove zero slices along the 1st dimension
def stripNullSlices(tmpMR,mrimg,labelimg):
	startS=-1
	endS=-1
	for i in range(0,tmpMR.shape[0]):
		if np.sum(tmpMR[i,:,:])<eps:
			if startS==-1:
				continue
			else:
				endS=i-1
				break
		else:
			if startS==-1:
				startS=i
			else:
				continue
	if endS==-1: #means there is no null slices at the end
		endS=tmpMR.shape[0]-1
	return startS,endS
    	
def main():
    #path='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
    #saveto='/shenlab/lab_stor3/dongnie/pelvicSeg/mrs_data/'
    path='/home/dongnie/warehouse/iSeg/iSeg-2017-Training/'
   
    ids=[1,2,3,4,5,6,7,8,9,10]
    #ids=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    #ids=[1,2,3,4,6,7,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    for ind in ids:
        dataT1filename='subject-%d-T1.hdr'%ind #provide a sample name of your filename of data here
        dataT2filename='subject-%d-T2.hdr'%ind #provide a sample name of your filename of data here
        dataT1fn=os.path.join(path,dataT1filename)
        dataT2fn=os.path.join(path,dataT2filename)
        labelfilename='subject-%d-label.hdr'%ind  # provide a sample name of your filename of ground truth here
        labelfn=os.path.join(path,labelfilename)
        imgT1Org=sitk.ReadImage(dataT1fn)
        mrT1img=sitk.GetArrayFromImage(imgT1Org)
        imgT2Org=sitk.ReadImage(dataT2fn)
        mrT2img=sitk.GetArrayFromImage(imgT2Org)
        #tmpMR=mrimg
       	#mrimg=mrimg

        labelOrg=sitk.ReadImage(labelfn)
        labelimg=sitk.GetArrayFromImage(labelOrg)
        labelimg[labelimg>200]=3 #white matter
        labelimg[labelimg>100]=2 #gray matter
        labelimg[labelimg>4]=1 #csf
        print 'labelimg elements are, ',np.unique(labelimg)
        #if ind<14:
		#	labelimg=labelimg/10 
        print np.unique(labelimg)
       	#mu=np.mean(labelimg)
       	#maxV, minV=np.percentile(labelimg, [99 ,1])
      	#labelimg=labelimg
       	#labelimg=(labelimg-mu)/(maxV-minV)
        #you can do what you want here for for your label img
        
        rate=1
        #print 'it comes to sub',ind
        print 'shape of mrimg, ',mrT1img.shape

        mu1=np.mean(mrT1img)
        mu2=np.mean(mrT2img)
		#maxV, minV=np.percentile(mrimg, [99 ,1])
        maxV1=np.ndarray.max(mrT1img)
        minV1=np.ndarray.min(mrT1img)
        maxV2=np.ndarray.max(mrT2img)
        minV2=np.ndarray.min(mrT2img)
        #print 'maxV1,',maxV1,' minV1, ',minV1
        std1=np.std(mrT1img)
        std2=np.std(mrT2img)
        #mrT1img=(mrT1img-mu1)/std1
        #mrT2img=(mrT2img-mu2)/std2
        mrT1img=(mrT1img-mu1)/(maxV1-minV1)
        mrT2img=(mrT2img-mu2)/(maxV2-minV2)
        print 'maxV1, ',np.ndarray.max(mrT1img),' maxV2, ',np.ndarray.max(mrT2img)
        #labelimg=labelimg[startS:endS+1,:,:]
        #originalLabel=labelimg
        #labelimg=labelimg[:,dim2_start:dim2_end,dim3_start:dim3_end]#attention region
        fileID='%d'%ind
        cubicCnt=extractPatches4OneSubject(mrT1img,mrT2img,labelimg,fileID,dFA,step,rate)
        print '# of patches is ', cubicCnt

        tmpMatT1=mrT1img
        tmpMatT2=mrT2img
        tmpLabel=labelimg
		#reverse along the 1st dimension 
        mrT1img=mrT1img[tmpMatT1.shape[0]-1::-1,:,:]
        mrT2img=mrT2img[tmpMatT2.shape[0]-1::-1,:,:]        
        labelimg=labelimg[tmpLabel.shape[0]-1::-1,:,:]
        fileID='%d_flip1'%ind
        cubicCnt=extractPatches4OneSubject(mrT1img,mrT2img,labelimg,fileID,dFA,step,rate)
		#reverse along the 2nd dimension 
        mrT1img=mrT1img[:,tmpMatT1.shape[1]-1::-1,:]
        mrT2img=mrT2img[:,tmpMatT2.shape[1]-1::-1,:] 
        labelimg=labelimg[:,tmpLabel.shape[1]-1::-1,:]
        fileID='%d_flip2'%ind
        cubicCnt=extractPatches4OneSubject(mrT1img,mrT2img,labelimg,fileID,dFA,step,rate)
		#reverse along the 2nd dimension 
        mrT1img=mrT1img[:,:,tmpMatT1.shape[2]-1::-1]
        mrT2img=mrT2img[:,:,tmpMatT2.shape[2]-1::-1]
        labelimg=labelimg[:,:,tmpLabel.shape[2]-1::-1]
        fileID='%d_flip3'%ind
        cubicCnt=extractPatches4OneSubject(mrT1img,mrT2img,labelimg,fileID,dFA,step,rate)
if __name__ == '__main__':     
    main()
