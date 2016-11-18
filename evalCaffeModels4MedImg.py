'''
Created on 11/01/2016
modified by Dong
'''
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes


# Make sure that caffe is on the python path:
caffe_root = '/usr/local/caffe3/'  # 
import sys
sys.path.insert(0, caffe_root + 'python')
print caffe_root + 'python'
import caffe
import h5py
import os
import SimpleITK as sitk
import scipy.io as scio


def postprocess(vol_out):
    r=int(vol_out.shape[1]/2.0)
    c=int(vol_out.shape[2]/2.0)
    sizecropup=150
    sizecropdown=100
    sizecrop=200

    mask=np.zeros_like(vol_out)
    mask[20:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=1
    mask[-25:,r-sizecropup:r+sizecropdown,c-sizecrop/2:c+sizecrop/2]=0
    vol_out*=mask
    print vol_out.shape
    print np.unique(vol_out)

    volheart=vol_out==2
    volheart=volheart.astype(np.uint8)
    idxheartini=np.where(volheart>0)

    volaorta=vol_out==4
    volaorta=volaorta.astype(np.uint8)
    idxaortaini=np.where(volaorta>0)

    voltrach=vol_out==3
    voltrach=voltrach.astype(np.uint8)
    idxtrachini=np.where(voltrach>0)

    voleso=vol_out==1
    voleso=voleso.astype(np.uint8)
    idxesoini=np.where(voleso>0)

    cc = sitk.ConnectedComponentImageFilter()

    vol_out1 = cc.Execute(sitk.GetImageFromArray(volheart))
    voltmp=sitk.RelabelComponent(vol_out1)
    volheartfiltered=sitk.GetArrayFromImage(voltmp)
    volheartfiltered=volheartfiltered==1

    vol_out2 = cc.Execute(sitk.GetImageFromArray(volaorta))
    voltmp=sitk.RelabelComponent(vol_out2)
    volaortafiltered=sitk.GetArrayFromImage(voltmp)
    volaortafiltered=volaortafiltered==1

    vol_out3 = cc.Execute(sitk.GetImageFromArray(voltrach))
    voltmp=sitk.RelabelComponent(vol_out3)
    voltrachfiltered=sitk.GetArrayFromImage(voltmp)
    voltrachfiltered=voltrachfiltered==1

    vol_out4 = sitk.GetImageFromArray(voleso)
    voltmp=sitk.BinaryMedian(vol_out4)
    volesofiltered=sitk.GetArrayFromImage(voltmp)

    maskheart=np.logical_and(volheartfiltered,volheart)
    maskaorta=np.logical_and(volaortafiltered,volaorta)
    masktrachea=np.logical_and(voltrachfiltered,voltrach)
    maskeso=volesofiltered>0#np.logical_and(volesofiltered,volesofiltered)

    for ind in xrange(volheartfiltered.shape[0]):
        maskheart[ind]=binary_fill_holes(maskheart[ind]).astype(int)
        maskaorta[ind]=binary_fill_holes(maskaorta[ind]).astype(int)
        masktrachea[ind]=binary_fill_holes(masktrachea[ind]).astype(int)
        maskeso[ind]=binary_fill_holes(maskeso[ind]).astype(int)
    idxheart=np.where(maskheart>0)
    idxaorta=np.where(maskaorta>0)
    idxtrachea=np.where(masktrachea>0)
    idxeso=np.where(maskeso>0)
    vol_out[idxheartini]=0
    vol_out[idxheart]=2
    vol_out[idxaortaini]=0
    vol_out[idxaorta]=4
    vol_out[idxtrachini]=0
    vol_out[idxtrachea]=3
    vol_out[idxesoini]=0
    vol_out[idxeso]=1

    volfinal=np.copy(vol_out)
    return volfinal

#this function is used to compute the dice ratio
def dice(im1, im2,organid):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """
    im1=im1==organid
    im2=im2==organid
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


# The code for run caffe, it looks like the global data, which ignores the main
# Load the net, list its data and params, and filter an example image.
caffe.set_device(1)
caffe.set_mode_gpu()


### load the solver and create train and test nets
solver = None  # ignore this workaround for lmdb data (can't instantiate two solvers on the same data)
#solver = caffe.SGDSolver('infant_fcn_solver.prototxt')
netjw = caffe.Net('cere_test.prototxt','cere_fcn_16x8_iter_300000.caffemodel',caffe.TEST)
print("blobs {}\nparams {}".format(netjw.blobs.keys(), netjw.params.keys()))

misblobs=netjw.blobs.keys()
for i in xrange(len(misblobs)):
    print misblobs[i]
    print netjw.blobs[misblobs[i]].data.shape 
    
path_patients = '/shenlab/lab_stor3/dongnie/mrs/'
#_, patients, _ = os.walk(path_patients).next()#every folder is a patient
mrfilename = 'img1.mhd'
segfilename = 'img1_seg.mhd'
ids = [1,2,3,4,6,7,8,10,11,12,13]
_, _, filename = os.walk(path_patients).next()#every file is a patient
volumes_dict={}
listdceso=[]
listdcheart=[]
listdcaorta=[]
listdctrachea=[]

listdceso_p=[]
listdcheart_p=[]
listdcaorta_p=[]
listdctrachea_p=[]

patients.sort()

patients_tmp=patients[1:8]#last 4 ones
for idx,namepatient in enumerate(patients_tmp):
    print namepatient
    ctitk=sitk.ReadImage(os.path.join(path_patients,namepatient,namepatient+'.mhd')) 
    ctnp=sitk.GetArrayFromImage(ctitk)
    ctnp[np.where(ctnp>3000)]=3000#we clap the images so they are in range -1000 to 3000  HU
    muct=np.mean(ctnp)
    stdct=np.std(ctnp)

    ctnp=(1/stdct)*(ctnp-muct)#normalize each patient

    segitk=sitk.ReadImage(os.path.join(path_patients,namepatient,'GT.nii.gz'))
    segnp=sitk.GetArrayFromImage(segitk)
    vol_out=np.zeros_like(segnp,dtype='uint8')

    for i in xrange(segnp.shape[0]):
        ctslice=ctnp[i,...]
        netroger.blobs['data'].data[0,0,...] = ctslice
        netroger.forward()
        out = netroger.blobs['softmax'].data[0].argmax(axis=0)
        vol_out[i]=out
        print 'slice {} done'.format(i)
    print namepatient


   

    dceso=dice(vol_out, segnp,1)
    dcheart=dice(vol_out, segnp,2)
    dctrachea=dice(vol_out, segnp,3)
    dcaorta=dice(vol_out, segnp,4)
    listdceso.append(dceso)
    listdcheart.append(dcheart)
    listdcaorta.append(dcaorta)
    listdctrachea.append(dctrachea)
    print 'eso {}'.format(dceso) 
    print 'heart {}'.format(dcheart)
    print 'trachea {}'.format(dctrachea)
    print 'aorta {}'.format(dcaorta)

    #np.save(namepatient+'_volout.npy',vol_out)

    
    print 'with postprocessing '
    vol_out=postprocess(vol_out)
    dceso=dice(vol_out, segnp,1)
    dcheart=dice(vol_out, segnp,2)
    dctrachea=dice(vol_out, segnp,3)
    dcaorta=dice(vol_out, segnp,4)
    listdceso_p.append(dceso)
    listdcheart_p.append(dcheart)
    listdcaorta_p.append(dcaorta)
    listdctrachea_p.append(dctrachea)
    print 'eso {}'.format(dceso) 
    print 'heart {}'.format(dcheart)
    print 'trachea {}'.format(dctrachea)
    print 'aorta {}'.format(dcaorta)

print 'Global normal'   
print 'mean eso ',np.mean(listdceso),'+- ',np.std(listdceso)
print 'mean heart ',np.mean(listdcheart),'+- ',np.std(listdcheart)
print 'mean trachea ',np.mean(listdctrachea),'+- ',np.std(listdctrachea)
print 'mean aorta ',np.mean(listdcaorta),'+- ',np.std(listdcaorta)

print 'Global with postprocessing'   
print 'mean eso ',np.mean(listdceso_p),'+- ',np.std(listdceso_p)
print 'mean heart ',np.mean(listdcheart_p),'+- ',np.std(listdcheart_p)
print 'mean trachea ',np.mean(listdctrachea_p),'+- ',np.std(listdctrachea_p)
print 'mean aorta ',np.mean(listdcaorta_p),'+- ',np.std(listdcaorta_p)

#def main():
#    # your code here
#
#if __name__ == "__main__":
#    # execute only if run as a script
#    main()
