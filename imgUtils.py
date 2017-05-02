'''
This script ensembles most of the function which is commonly used in medical image
'''
import numpy as np
import h5py

def psnr(gtMat,preMat):
    print preMat.shape
    print gtMat.shape

    mse=np.sqrt(np.mean((gtMat-preMat)**2))
    print 'mse ',mse
    max_I=np.max([np.max(gtMat),np.max(preMat)])
    print 'max_I ',max_I
    return 20.0*np.log10(max_I/mse)


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

class ScanFile(object):   
    def __init__(self,directory,prefix=None,postfix=None):  
        self.directory=directory  
        self.prefix=prefix  
        self.postfix=postfix  
          
    def scan_files(self):    
        files_list=[]    
            
        for dirpath,dirnames,filenames in os.walk(self.directory):   
            ''''' 
            dirpath is a string, the path to the directory.   
            dirnames is a list of the names of the subdirectories in dirpath (excluding '.' and '..'). 
            filenames is a list of the names of the non-directory files in dirpath. 
            '''  
            for special_file in filenames:    
                if self.postfix:    
                    special_file.endswith(self.postfix)    
                    files_list.append(os.path.join(dirpath,special_file))    
                elif self.prefix:    
                    special_file.startswith(self.prefix)  
                    files_list.append(os.path.join(dirpath,special_file))    
                else:    
                    files_list.append(os.path.join(dirpath,special_file))    
                                  
        return files_list    
      
    def scan_subdir(self):  
        subdir_list=[]  
        for dirpath,dirnames,files in os.walk(self.directory):  
            subdir_list.append(dirpath)  
        return subdir_list  
  
if __name__=="__main__":  
    dir=r"D:xxxx"  
    scan=ScanFile(dir)  
    subdirs=scan.scan_subdir()  
    files=scan.scan_files()  
      
    print "The subdirs scaned are:"  
    for subdir in subdirs:  
        print subdir  
      
    print "The files scaned are:"  
    for file in files:  
        print file  


def main():
    dataPath='D:\unc\bric\data\MRI_CT_histmatch_brain_prostate\fcn_bnRes\res_s16_v13/'
    ids=[1,2,3,4,5,6,7,8,9,10,11]
    for id in ids:
        gtfn='gt%d_s16.nii'%id
        prefn='preSub%d_s16_v13.nii'%id
        gtOrg=sitk.ReadImage(dataPath+gtfn)
        gtMat=sitk.GetArrayFromImage(gtOrg)
        preOrg=sitk.ReadImage(dataPath+prefn)
        preMat=sitk.GetArrayFromImage(preOrg)
        pr=psnr(gtMat,preMat)
        print 'psnr for %d is %f'%id,pr
        
if __name__ == '__main__':     
    main()
