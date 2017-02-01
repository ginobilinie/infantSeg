import sys  
import glob  
import os  
import SimpleITK as sitk

dir = './'  
suffix = 'nii'
f = glob.glob(dir + '*.' + suffix)  
for file in f :  
    filename = os.path.basename(file)  
    print filename
    labelOrg=sitk.ReadImage(filename)
    labelimg=sitk.GetArrayFromImage(labelOrg)
    
    gtVol=sitk.GetImageFromArray(labelimg)
    sitk.WriteImage(gtVol,filename+'.gz')
 