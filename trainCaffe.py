import numpy as np
import caffe
import os
import simpleITK as sitk
import h5py
import caffe.draw
import subprocess
import platform
import google.protobuf 
import google.protobuf 

caffe_root='/usr/local/caffe3/'

caffe.set_device(2) #very important
caffe.set_mode_gpu()

solverPrototxtFilename = '/your/path/to/your_solver.prototxt'
trainPrototxtFilename = '/your/path/to/your_train_test.prototxt'
deployPrototxtFilename  = '/your/path/to/your_deploy.prototxt'
trainDataH5Filename = '/your/path/to/your_train_data.hdf5' 
testDataH5Filename = '/your/path/to/your_test_data.hdf5' 
caffemodel_filename='/your/path/to/your_trained_model.caffemodel' #your trained caffemodel
'''
you can define your own loadData function, please
'''
def loadData():
    '''
    use simpleITK to read your data firstly, and then 
    '''
    #define your read data codes sections
    #read input as data
    #read targets as labels
    new_data = {}
    #new_data['input'] = data.data
    new_data['input'] = np.reshape(data, (batch_size,channels,width,height,depth))
    new_data['output'] = labels

    return new_data

def convertData2Hdf5(hdf5_data_filename, data):
    '''
    convert you data (ndarray form) to hdf5 files
    '''
    with h5py.File(hdf5_data_filename, 'w') as f:
        f['data'] = data['input'].astype(np.float32)
        f['label'] = data['output'].astype(np.float32)

def train(solverPrototxtFilename):
    '''
    Train the CNN model
    '''
    solver = caffe.get_solver(solverPrototxtFilename)
    solver.solve()

def printNetParams(net):
    '''
    Print the parameters of the network
    '''
    print(net)
    print('net.inputs: {0}'.format(net.inputs))
    print('net.outputs: {0}'.format(net.outputs))
    print('net.blobs: {0}'.format(net.blobs))
    print('net.params: {0}'.format(net.params))  
    

def printNetwork(prototxt_filename, caffemodel_filename):
    '''
    Draw the CNN architecture as you want, actually, it may not work property
    '''
    _net = caffe.proto.caffe_pb2.NetParameter()
    f = open(prototxt_filename)
    google.protobuf.text_format.Merge(f.read(), _net)
    caffe.draw.draw_net_to_file(_net, prototxt_filename + '.png' )
    print('Draw CNN finished!')

def printNetwork_weights(prototxt_filename, caffemodel_filename):
    '''
    For each CNN layer, print weight heatmap and weight histogram 
    '''
    net = caffe.Net(prototxt_filename,caffemodel_filename, caffe.TEST)
    for layerName in net.params: 
        # display the weights
        arr = net.params[layerName][0].data
        plt.clf()
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        cax = ax.matshow(arr, interpolation='none')
        fig.colorbar(cax, orientation="horizontal")
        plt.savefig('{0}_weights_{1}.png'.format(caffemodel_filename, layerName), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()

        # weights histogram  
        plt.clf()
        plt.hist(arr.tolist(), bins=20)
        plt.savefig('{0}_weights_hist_{1}.png'.format(caffemodel_filename, layerName), dpi=100, format='png', bbox_inches='tight') # use format='svg' or 'pdf' for vectorial pictures
        plt.close()
        

def main():
 # Set parameters

    # Prepare data
    data = loadData()
    convertData2Hdf5(trainDataH5Filename, data)
    #convertData2Hdf5(testDataH5Filename, data)

    # Train network
    train(solverPrototxtFilename)

    # Print network
#     printNetwork(deployPrototxtFilename, caffemodel_filename)
#     printNetwork(trainPrototxtFilename, caffemodel_filename)
#     printNetwork_weights(trainPrototxtFilename, caffemodel_filename)

    #to evaluate your trained model, you can use the 'evalCaffeModel4WholeImgNieSingleS.py' file

if __name__ == '__main__':     
    main()


