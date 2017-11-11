
# coding: utf-8

# # This script obtaining frames from camera,using mtcnn detecting faces,croping and embedding faces with pre-trained facenet and finally face recogition with pre-trained classifier.
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from os.path import join as pjoin
import sys
import copy
import detect_face
import nn4 as network
import random
import sklearn

from sklearn.externals import joblib

#face detection parameters
minsize = 20 # minimum size of face
threshold = [ 0.5, 0.6, 0.7 ]  # three steps's threshold
factor = 0.5 # scale factor

#facenet embedding parameters

model_dir='./model_check_point/model.ckpt-500000'#"Directory containing the graph definition and checkpoint files.")
model_def= 'models.nn4'  # "Points to a module containing the definition of the inference graph.")
image_size=96 #"Image size (height, width) in pixels."
pool_type='MAX' #"The type of pooling to use for some of the inception layers {'MAX', 'L2'}.
use_lrn=False #"Enables Local Response Normalization after the first layers of the inception network."
seed=42,# "Random seed."
batch_size= None # "Number of images to process in a batch."



frame_interval=1 # frame intervals  


# In[3]:

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret

#restore mtcnn model
print('Creating networks and loading parameters')
gpu_memory_fraction=1.0
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')

print('Creating networks completely')

import facenet

# restore facenet model
print('facenet embedding')

with tf.Graph().as_default():
    with tf.Session() as sess:
        model_path='./model_check_point/20170512-110547/20170512-110547.pb'
        # model_path = model_dir
        print('Loading feature extraction model')
        facenet.load_model(model_path)
        
        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0") # 128 nodes
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]


print('Loading model completely')

#restore pre-trained knn classifier
# model = joblib.load('./model_check_point/knn_classifier.model')


# # real time face detection and recognition

#obtaining frames from camera--->converting to gray--->converting to rgb
#--->detecting faces---->croping faces--->embedding--->classifying--->print

import cv2

video_capture = cv2.VideoCapture(0)
c=0
frame_counter = 0
import time
# ret, frame = video_capture.read()
current_time = time.time()
print current_time
while(video_capture.isOpened()):  # check!
    start = time.time()
    # Capture frame-by-frame
    ret, frame = video_capture.read()
        
    #==============================================#

    timeF = frame_interval

    if ret == False:
        continue
    
    # if(c % timeF == 0): #frame_interval==3, face detection every 3 frames
    if True:
        # find_results=[]
        
        # our operation on frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
               
        if gray.ndim == 2:
            img = to_rgb(gray)
        
        bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
        
        nrof_faces = bounding_boxes.shape[0]#number of faces
        

        for face_position in bounding_boxes:
            face_position=face_position.astype(int)
            
            
            cv2.rectangle(frame, (face_position[0], 
                            face_position[1]), 
                            (face_position[2], face_position[3]), 
                            (0, 255, 0), 2)
            
            # frame1 = frame
            # crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]
    
            # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
            # crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )
        
            # data=crop.reshape(-1,96,96,3)
            # data=crop.reshape(-1, 160, 160, 3)
        
            # feed_dict = {images_placeholder: data, phase_train_placeholder: False}
            # emb_data = sess.run(embeddings, feed_dict = feed_dict)[0]

    
        # Draw a rectangle around the faces
        # cv2.putText(frame,'detected:{}'.format(find_results), (50,100), 
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0 ,0), 
        #             thickness = 2, lineType = 2)
      
            
    #print(faces)
    c+=1    
    
    # Display the resulting frame
    #cv2.imshow('frame', gray)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    print('caculating in ', time.time() - start)  

# When everything is done, release the capture

video_capture.release()
cv2.destroyAllWindows()


