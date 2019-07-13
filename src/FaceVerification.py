'''
An example to show the interface.
'''
#from skimage import io
#from somewhere import load_model
import cPickle as pickle
import caffe
import numpy as np
import copy
# Note to load your model outside of `FaceVerification` function,
# otherwise, model will be loaded every comparison, which is too time-consuming.


#load trained deepid model
model_def = "../model/deepid_deploy.prototxt"         # defines the structure of the model
model_weights = "../model/snapshot_iter_400000.caffemodel" # contains the trained weights
mean_file = "../model/train_mean.npy"
net = caffe.Net(model_def,      
                model_weights,  
                caffe.TEST) 

mu = np.load(mean_file)
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values


#load learned A & G for JointBayesian 
model_fold="../model/"
with open(model_fold+"A.pkl", "r") as f:
    A = pickle.load(f)
    f.close
with open(model_fold+"G.pkl", "r") as f:
    G = pickle.load(f)
    f.close


    
#get a 160-dimension vector as feature of a single image 
def get_feature(imagepath):  
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
	
    net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          64, 64)  # image size is 227x227
						   
    image = caffe.io.load_image(imagepath)
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
	
    output = net.forward()
    
    output_fc160 = output['fc160'][0]
    return output_fc160

#compute the corrolation of two image features by using JointBayesian
def JointBayesian_Verify(x1, x2): 
    #transform to be a column vector 
    x1.shape = (-1,1) 
    x2.shape = (-1,1)
	
	#compute the corrolation of two image features by using JointBayesian
    ratio = np.dot(np.dot(np.transpose(x1),A),x1) + np.dot(np.dot(np.transpose(x2),A),x2) - 2*np.dot(np.dot(np.transpose(x1),G),x2)
    
    return float(ratio)
    
    

def FaceVerification(img_path1, img_path2):  
    
    tmp=get_feature(img_path1)
    f1=copy.deepcopy(tmp)
    tmp=get_feature(img_path2)
    f2=copy.deepcopy(tmp)
    
    ratio=JointBayesian_Verify(f1, f2)
   
    #the threshold is selected by doing experiments
    threshold=-7.005
    
    if ratio > threshold:
		return 1
    else:
		return 0
  