import numpy as np
import h5py
import SimpleITK as sitk

filename='trainMS_1.h5'
with h5py.File(filename) as hf:
    dataMR=hf.get('dataMR')
    npdataMR=np.array(dataMR)
    print npdataMR.shape
    dataCT=hf.get('dataCT')
    npdataCT=np.array(dataCT)
    print npdataCT.shape
    patchMR=npdataMR[20,0,:,:,:]
    patchCT=npdataCT[20,0,:,:,:]
    patchMRVol=sitk.GetImageFromArray(patchMR)
    patchCTVol=sitk.GetImageFromArray(patchCT)
    sitk.WriteImage(patchMRVol,'patchMR.nii')
    sitk.WriteImage(patchCTVol,'patchCT.nii')
