    
'''
Target: avg for kinds of medical images, such as hdr, nii, mha, mhd, raw and so on, and store them as hdf5 files
Created on Oct. 20, 2016
Author: Dong Nie 
'''



import SimpleITK as sitk

from multiprocessing import Pool
import os
import h5py
import numpy as np  

def main():
    path='./'
    saveto='./'
   
    datafilename='Brain_1to1_CT_resampled_from2to1.hdr' #provide a sample name of your filename of data here
    datafn=os.path.join(path,datafilename)
    imgOrg=sitk.ReadImage(datafn)
    img1=sitk.GetArrayFromImage(imgOrg)
    
    labelfilename='Brain_1to1_CT_resampled_from3to1.hdr'  # provide a sample name of your filename of ground truth here
    labelfn=os.path.join(path,labelfilename)
    imgOrg=sitk.ReadImage(datafn)
    img2=sitk.GetArrayFromImage(imgOrg)
    
    labelfilename='Brain_1to1_CT_resampled_from4to1.hdr'  # provide a sample name of your filename of ground truth here
    labelfn=os.path.join(path,labelfilename)
    imgOrg=sitk.ReadImage(datafn)
    img3=sitk.GetArrayFromImage(imgOrg)
    
    img=(img1+img2+img3)/3
    
    
    volOut=sitk.GetImageFromArray(img)
    sitk.WriteImage(volOut,'preSub1_MV.nii')

if __name__ == '__main__':     
    main()
