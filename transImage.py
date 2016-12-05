'''
Target: Transpose (permute) the order of dimensions of a image  
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio

def main():
    ids=[1,2,3,4,5,6,7]
    for id in ids:
        gtfn='gt%d_as32.nii'%id
        gtOrg=sitk.ReadImage(gtfn)
        gtMat=sitk.GetArrayFromImage(gtOrg)
        gtMat=np.transpose(gtMat,(2,1,0))
        gtVol=sitk.GetImageFromArray(gtMat)
        sitk.WriteImage(gtVol,gtfn)
        
        prefn='preSub%d_as32_v12.nii'%id
        preOrg=sitk.ReadImage(prefn)
        preMat=sitk.GetArrayFromImage(preOrg)
        preMat=np.transpose(preMat,(2,1,0))
        preVol=sitk.GetImageFromArray(preMat)
        sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
