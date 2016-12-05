# infantSeg

This is the project for multi-modality infant brain segmentation at the isointense stage. We develop deep learning based methods to deal with the segmentation task.
More will be uploaded after the journal paper is accepted.

evalCaffeModels4MedImg.py: this is the code to evaluate a whole 3d image on the trained DL models for the patients. 

readMedImg4CaffeCropNie.py: this is the code to extract 3d patches from a medical image (mri/ct and so on) for patients. Note, I am not that sure if this contains error or not (I wrote too many versions, I donot remember which version it is), if it contains, please let me know by QQ. 

readMedImg4CaffeCropNie4SingleS.py: implement the crop patches function, and this one is much better than readMedImg4CaffeCropNie.py, so I suggest you use this one.

evalCaffeModel4ImgNie.py: this is the code to evaluate a whole 3d image on the trained DL models for the patients (suggested using this one).

transData.py: this is written to transpose (permute in matlab) the dimension order of the h5 (hdf5 in matlab).

transImage.py: this is written to transpose (permute in matlab) the dimension order of a medical image.

infant_train_test.prototxt: define the network architecture

infant_solver.prototxt: define the network hyperparameters

Basic steps:

1. crop patches from medical image data with readMedImg4CaffeCropNie4SingleS.py, I use SimpleITK as readers.

2. prepare the network: define infant_train_test.prototxt, and decide infant_solver.prototxt

3. train the network with caffe

4. evaluate the caffe model with evalCaffeModel4ImgNie.py, I store them in nifti format.


