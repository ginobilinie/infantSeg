'''
    Transpose (permute the data)
'''
import numpy as np
import h5py

def main():
    ids=[1,2,3,4,5,6,7,8,9,10,11]
    for id in ids:
        filename='trainMS_%d.h5'%id
        with h5py.File(filename) as hf:
            dataMR8=hf.get('dataMR8')
            dataSeg8=hf.get('dataSeg8')
            dataMR16=hf.get('dataMR16')
            dataSeg16=hf.get('dataSeg16')
            dataMR32=hf.get('dataMR32')
            dataSeg24=hf.get('dataSeg24')
            
            np_dataMR8=np.array(dataMR8)
            dataMR8=np.transpose(np_dataMR8,(4,3,2,1,0))
            np_dataSeg8=np.array(dataSeg8)
            dataSeg8=np.transpose(np_dataSeg8,(4,3,2,1,0))
            
            np_dataMR16=np.array(dataMR16)
            dataMR16=np.transpose(np_dataMR16,(4,3,2,1,0))
            np_dataSeg16=np.array(dataSeg16)
            dataSeg16=np.transpose(np_dataSeg16,(4,3,2,1,0))

            np_dataMR32=np.array(dataMR32)
            dataMR32=np.transpose(np_dataMR32,(4,3,2,1,0))
            np_dataSeg24=np.array(dataSeg24)
            dataSeg24=np.transpose(np_dataSeg24,(4,3,2,1,0))

            with h5py.File('./trainMS_%s.h5'%id,'w') as f:
            f['dataMR32']=dataMR32
            f['dataSeg24']=dataSeg24
            f['dataMR16']=dataMR16
            f['dataMR8']=dataMR8
            f['dataSeg16']=dataSeg16
            f['dataSeg8']=dataSeg8
            with open('./trainMS_list.txt','a') as f:
                f.write('./trainMS_%s.h5\n'%id)
               
if __name__ == '__main__':
    main()

