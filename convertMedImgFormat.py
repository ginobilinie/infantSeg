'''
Target: Transpose (permute) the order of dimensions of a image  
Created on Jan, 22th 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio

path='highresCTMR_29/'
def main():
    ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    for id in ids:
        gtfn=os.path.join(path,'p%d/alignmr.mhd'%id)
        outfn=os.path.join(path,'p%d/alignmr.nii'%id)
        gtOrg=sitk.ReadImage(gtfn)
        gtMat=sitk.GetArrayFromImage(gtOrg)
        #gtMat=np.transpose(gtMat,(2,1,0))
        gtVol=sitk.GetImageFromArray(gtMat)
        sitk.WriteImage(gtVol,outfn)
#         
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
