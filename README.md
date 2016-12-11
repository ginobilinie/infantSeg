# infantSeg

This is the project for multi-modality infant brain segmentation at the isointense stage. We develop deep learning based methods to deal with the segmentation task.
More will be uploaded after the journal paper is accepted.

Basic steps:

1. crop patches from medical image data with readMedImg4CaffeCropNie4SingleS.py, I use SimpleITK as readers.

2. prepare the network: define infant_train_test.prototxt, and decide infant_solver.prototxt. And if your network is convolution-free (which I mean is MLP), you can refer to my another proj: https://github.com/ginobilinie/psychoTraitPrediction 

3. train the network with caffe

4. evaluate the caffe model with evalCaffeModel4ImgNie.py, I store them in nifti format.

If you think it is useful for you, please star it. HAHA. And you can cite this paper (this paper is a simplified version of our paper, a better version is under review by a journal, I'll notify you if it is AC):</br>
"Nie, Dong, et al. "Fully convolutional networks for multi-modality isointense infant brain image segmentation." Biomedical Imaging (ISBI), 2016 IEEE 13th International Symposium on. IEEE, 2016."


.........................................................................................................................................................................................................................................................

<B>Here is a list of what the network related files mean, and for the below thre prototxt, I'd like to share some basic experience how to write them and where we need be careful:</B>

a. infant_train_test.prototxt: define the network architecture</br>
The architecture actually is not hard to define, </br>
number of total layers: the number of layers could be easily initially decided, you can just make the receptive field for the last layer as large as the input. Of course, you should search what receptive field is, and I can give you a suggestion one to read: https://www.quora.com/What-is-a-receptive-field-in-a-convolutional-neural-network.</br>
convolution paramter settings: just see my example (infant_train_test.prototxt), and you can improve it based on this, for example, you can use other initialization methods (I like to use Xavier, you can use others) </br>
activation function: ReLU</br>
fully connected layer setting: refer to prototxt in https://github.com/ginobilinie/psychoTraitPrediction . </br>
....

b. infant_solver.prototxt: define the network hyperparameters
The most important parameters you should take care is learning rate (lr), and learning rate decay strategy (learning_policy: fixed or step).  stepsize( this is necessary when you set the learning policy as step, which means it will decrease gamma times when it reaches every stepsize steps).</br>
momutum: you can set by 0.9 as default.</br>
maxIter should be the maximum iterations for the network to train, usually you can set two times as large as you dataset.</br>
And if you want to use other optimization instead of SGD, you have to set 'type', for example, type: "Adam".</br>

c. infant_deploy.prototxt: the deploy prototxt when you want to evaluate your trained caffe model.
You only have to make two changes based on the train_test.prototxt: 
input (replace the original input (e.g., HDF5) with the dimension described format which I have a example in infant_deploy.prototxt.</br>
and also, you have to replace to output softmaxWithLoss with softmax layer.</br>

d. infant_train_test_finetune.prototxt: finetune prototxt when you want to use previous model the initialize the training of a new model.</br>
Notice that we should keep almost everything (instead of learning rate) of the layers you like to initialize the same between the old trained model and the new model, and then you can add new layers in the new model. compare the infant_train_test_finetune.prototxt and infant_train_test.prototxt, you will know how it works.

<B>How to train/resume/finetune:</B>

a. train_infant.sh: shell code to train the model

b. resume_infant.sh: resume a previous trained model</br>
If your training is broken by something else, then you should not train from scratch, instead, you can resume the training.

c. finetune_infant.sh: finetune a previous trained model</br>
If you can have more dataset, and then you'd like to finetune the previous trained model, and then you can finetune the trained model. </br>
If you like to train a new model, and you like to fix the first several layers's weights, just train some layers or some new layers, you can also use finetune. In this case, I have edited a infant_train_test_finetune.prototxt, in which the learning rate of the fixing layers is set to 0 (then you can fix these layers' weights, and then finetue it. 
Just use finetune_infant.sh to run it. 
...........................................................................................................................................................................................................................................................

<B>Here are some codes you may want to use: If you have no IDE for python, I suggestion you use Eclipse+pyDev as editor and compiler for python codes:</B>

a. evalCaffeModels4MedImg.py: this is the code to evaluate a whole 3d image on the trained DL models for the patients. 

b. readMedImg4CaffeCropNie.py: this is the code to extract 3d patches from a medical image (mri/ct and so on) for patients. Note, I am not that sure if this contains error or not (I wrote too many versions, I donot remember which version it is), if it contains, please let me know by QQ. 

c. readMedImg4CaffeCropNie4SingleS.py: implement the crop patches function, and this one is much better than readMedImg4CaffeCropNie.py, so I suggest you use this one.

d. evalCaffeModel4ImgNie.py: this is the code to evaluate a whole 3d image on the trained DL models for the patients (suggested using this one). And if you want to read the intermedia layer's output, you can specify it by "temppremat = mynet.blobs['layername'].data[0]".

e. transData.py: this is written to transpose (permute in matlab) the dimension order of the h5 (hdf5 in matlab). Actually, you can also check the hdf5 format data if the label and feature match or not...

f. transImage.py: this is written to transpose (permute in matlab) the dimension order of a medical image.

g. convertMat2MedFormat.py: convert mat files (matlab) to medical image format, such as nifti, hdr and so on

h. checkHDF5.py: check the hdf5 files you generated to find if there are something wrong (e.g., matched or not)

