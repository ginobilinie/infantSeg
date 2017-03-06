'''
Target: Transpose (permute) the order of dimensions of a image  
Created on March, 5th, 2017
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  
import scipy.io as scio
from PIL import Image
import scipy.misc as scmi
from scipy import ndimage as nd


path='/shenlab/lab_stor5/shzhou16/CuttedData192/planct1/'
outPath='/shenlab/lab_stor5/yzgao/planct1/'
def main():
    #ids=[14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
    ids=range(1,41)
    rate=2.0/3
    
    for id in ids:
        gtfn=os.path.join(path,'P%d/V2Rct.nii.gz'%id)
#         outfn=os.path.join(path,'P%d/V2Rct_all.nii.gz'%id)
        gtOrg=sitk.ReadImage(gtfn)
        gtMat=sitk.GetArrayFromImage(gtOrg)
        print 'mat shape, ', gtMat.shape
        for s in range(1,gtMat.shape[3]):
            sliceMat=gtMat[:,:,s]
            sliceMatScale = nd.interpolation.zoom(sliceMat, zoom=rate)
            scmi.imsave('p%d_'%id+'s%d.png'%s, sliceMat)
        #gtMat=np.transpose(gtMat,(2,1,0))
#         gtVol=sitk.GetImageFromArray(gtMat)
#         sitk.WriteImage(gtVol,outfn)
#         
#         prefn='preSub%d_as32_v12.nii'%id
#         preOrg=sitk.ReadImage(prefn)
#         preMat=sitk.GetArrayFromImage(preOrg)
#         preMat=np.transpose(preMat,(2,1,0))
#         preVol=sitk.GetImageFromArra(preMat)
#         sitk.WriteImage(preVol,prefn)
 
        
if __name__ == '__main__':     
    main()
