'''
Written to convert mat format file (matlab) to medical format (nii, hdr, mhd and so on)
Dong Nie
'''
import SimpleITK as sitk
from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from scipy.io import loadmat

path=''
def main():
    dataPath='/home/dongnie/caffe3D/python/prostate/fcnBNRes32to16/'
    ids=[1,2,3,4,5,6,7,8,9,10,11]
    for ind in ids:
    	prefName='pre3dFCN32to16_%d.mat'%ind
    	x = loadmat(prefName)
    	preMat=x['preMat']
    	gtMat=x['gtMat']
    	prefn='preSub%d_s32_32to16.nii'%ind
        preVol=sitk.GetImageFromArray(preMat)
        sitk.WriteImage(preVol,prefn)
        
        gtfn='gt%d_s32.nii'%ind
 		segVol=sitk.GetImageFromArray(gtMat)
        sitk.WriteImage(segVol,prefn)
        
        pr=psnr(gtMat,preMat)
        print 'psnr for %d is %f'%id,pr
        
if __name__ == '__main__':     
    main()
