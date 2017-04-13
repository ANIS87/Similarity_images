
import numpy as np
import sys
import os 
import h5py
import math
import string
import sys, getopt
sys.path.append("/home/caffe/python")
import caffe
caffe.set_mode_gpu()
import subprocess
from scipy.spatial import distance
#function

def convert_h5_to_array(h5_file):
   f = h5py.File(h5_file, 'r')   
   a_group_key = list(f.keys())
   print(a_group_key)
   # Get the data
   data = list(f[a_group_key[0]])
   #print(data[0])
   return data[0]

#global code


class siamese_similarity():
   #global parameter of network similarity
   def __init__(self, caffe_model, caffe_deploy, mean_file,data_base_retrival):
       self.train_siame=caffe_model
       self.network_siame=caffe_deploy
       self.retrieval_base=data_base_retrival
       self.mean_file=mean_file
       mean_data = np.load(self.mean_file)
       self. mean_data = mean_data.mean(1).mean(1)
  #extarct one barnche
   def get_one_branch(self):
      simnet = caffe.Net(self.network_siame,self.train_siame, caffe.TEST)
      name_model=str.replace(self.train_siame, 'model/','')
      self.one_branch='model/one_branch_'+name_model
      simnet.save(self.one_branch)
      return self.one_branch

  #preparation of data base retrieval
      
   def compute_features(self):
     #read list retrievla images base form a .txt file 
     f = open(self.retrieval_base, 'r')
     list_images = list(f)
     f.close()
     i = 1
     print (str(len(list_images)) + " images")  
        
     ## Loading net (one branch to simplify the speed up the calculation)
     #cheek if one brach is calculted or no
     if os.path.exists(self.one_branch):
        pass
     else:
        self.one_branch=self.get_one_branch()
 
     #run data base preparation "features extractions"
     self.net_exfeat = caffe.Net(self.network_siame,self.one_branch, caffe.TEST)

     # create transformer for the input called 'data'
     transformer = caffe.io.Transformer({'data': self.net_exfeat.blobs['data'].data.shape})
     transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
     transformer.set_mean('data', self.mean_data)        # subtract the dataset-mean value in each channel
     transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
     transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    
     for im_query in list_images:
        
        self.net_exfeat.blobs['data'].reshape(1, 3, 224, 224)
        print( str(i) +  " / " + str(len(list_images)) + " : " +  im_query[:-1])        
        image = caffe.io.load_image(im_query[:-1])
        data_query = transformer.preprocess('data', image)

        # Extract feature
        self.net_exfeat.blobs['data'].data[...] = data_query
        self.net_exfeat.forward()

        data_p = self.net_exfeat.blobs['norm_data'].data
        data_p = np.array(data_p)
        
        h5_file=str(os.path.splitext(im_query)[0]+'.h5')
        with h5py.File(h5_file, 'w') as hf:
            hf.create_dataset('data', data=data_p)

        i = i + 1
     return self.net_exfeat
   

   def compute_feature_single_image(self, user_image):
      if os.path.exists(self.one_branch):
        pass
      else:
        self.one_branch=self.get_one_branch()
     
      my_model = caffe.Net(self.network_siame,self.one_branch, caffe.TEST)
      # create transformer for the input called 'data'
      transformer = caffe.io.Transformer({'data': my_model.blobs['data'].data.shape})
      transformer.set_transpose('data', (2,0,1))     # move image channels to outermost dimension
      transformer.set_mean('data', self.mean_data)        # subtract the dataset-mean value in each channel
      transformer.set_raw_scale('data', 255)         # rescale from [0, 1] to [0, 255]
      transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
      my_model .blobs['data'].reshape(1, 3, 224, 224)
      image_test=caffe.io.load_image(user_image)
      data_test= transformer.preprocess('data', image_test)
      # Extract feature
      my_model.blobs['data'].data[...] = data_test
      my_model.forward()

      feat_test = my_model.blobs['norm_data'].data
      feat_test = np.array(feat_test)
      return feat_test

   def find_similiar_product(self,image_test, data_base_prepared='True'):
       all_distance=[]
       if  data_base_prepared:
           pass
       else:
         self.net_exfeat=self.compute_features()
       my_test_data=self.compute_feature_single_image(image_test)
       #read list retrievla images base form a .txt file 
       f = open(self.retrieval_base, 'r')
       list_features = list(f)
       f.close()
       j=0
       for ret_feat in list_features:
          feat_h5_file=os.path.splitext(ret_feat)[0]+'.h5'
          print('hh', feat_h5_file)
          data_ret_feat=convert_h5_to_array(feat_h5_file)
          d_test_feat=distance.euclidean(data_ret_feat, my_test_data)
          all_distance.insert(j, d_test_feat)
          j=j+1
       #find similair images
       all_dist = np.array(all_distance).astype(float)
       # print(all_dist)
       best = all_dist.argsort()
       # print 2 best product
       number_to_select=2
       very_sim = best[0:number_to_select]  
       images_sim=[list_features[t] for t in  very_sim]
       print(all_dist)
       
       return  images_sim 
############################################
#use
###########################################
caffe_model='model/_iter_90000.caffemodel' #
caffe_deploy='model/Siamese_VGG_16_layers_2_deploy.prototxt' #same file (not changed)
mean_file='model/ilsvrc_2012_mean.npy' #to noramlize image
data_base_retrival='retrival_data_base.txt'  #lis of retrival image "sarenza" forma ".jpg"  or ".jpeg" 

#initalization model
my_siam_network=siamese_similarity( caffe_model, caffe_deploy, mean_file,data_base_retrival)
#extarct one  branch
my_siam_network.get_one_branch()
#data base features extraction 
my_siam_network.compute_features() #result file ".h5" saved in the folder retrival ! ! ! (not change the folder)
#similarity calcul
image_test='test.jpg'
similair_images=my_siam_network.find_similiar_product(image_test)
print(similair_images) #two best similair image
