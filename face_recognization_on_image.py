
# coding: utf-8

# This script obtaining frames from camera,using mtcnn detecting faces,croping 
# and embedding faces with pre-trained facenet and finally face recogition with 
# pre-trained classifier.
from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import cv2
from os.path import join as pjoin
import sys
import copy
import detect_face
import random
import sklearn
from sklearn.externals import joblib
import argparse
import facenet
import math

def to_rgb(img):
    """Convert from gray 1 channel to gray 3 channels
    """
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def main():

    #restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction=0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default(): 
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
    print('Creating networks completely')


    with tf.Graph().as_default():
        with tf.Session() as sess:
            # Face detection parameters
            minsize = 20 # minimum size of face
            threshold = [ 0.5, 0.6, 0.7 ]  # three steps's threshold
            factor = 0.7 # scale factor
            
            # Load the model
            print('Loading feature extraction model')
            model_path = './model_check_point/20170512-110547.pb'
            facenet.load_model(model_path)   

            # Get input and output tensors
            print('Get input and output tensors')
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Restore pre-trained knn classifier
            model, class_names = joblib.load('./models/face_classifer-v6.pkl')

            # Process with every single face
            print('class_names:  ', class_names )
                    
            import cv2
            import time
            start = time.time()

            # Loading image
            image = cv2.imread(args.img_dir)

            # Converting to gray
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            if gray.ndim == 2:
                img = to_rgb(gray)

            bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

            nrof_faces = bounding_boxes.shape[0] # number of faces
                        

            for index, face_position in enumerate(bounding_boxes):
                print("-----------------------------------------\nFace #" + str(index + 1) + ":")
                face_position=face_position.astype(int) 

                cv2.rectangle(image, (face_position[0], 
                                face_position[1]), 
                                (face_position[2], face_position[3]), 
                                (0, 255, 0), 2)
                
                cv2.rectangle(image, 
                            (face_position[0], face_position[1] - (face_position[2] - face_position[0])/4), 
                            (face_position[2], face_position[1]), 
                            (0, 255, 0), -1)
                
                
                crop=img[face_position[1]:face_position[3],face_position[0]:face_position[2],]

                # # crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_CUBIC )
                crop = cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC )

                # # data=crop.reshape(-1,96,96,3)
                crop = crop.reshape(-1, 160, 160, 3)

                # Run forward pass to calculate embeddings
                image_size = 160
                print('Calculating features for images')
                images = facenet.process_image(crop, False, False, image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_data = sess.run(embeddings, feed_dict=feed_dict)

                predictions = model.predict_proba(emb_data).ravel()

                best_class_indices = np.argmax(predictions)
                confidence = predictions[best_class_indices]

                print("Confidence: ", confidence)
                print("Predicting probabilties: ", predictions)
                
                if confidence > 0.5:
                    person = str(class_names[best_class_indices])
                    print("Predict: ", class_names[best_class_indices])
                else:
                    person = "Unknown"
                    print("Predict: Unknown")

                # Drawing predicting box
                text_tl = (face_position[0], face_position[1] - (face_position[2] - face_position[0])/12)
                text_scale = (face_position[2] - face_position[1])*1.5
                # text_br = (int(face_position[2]), int(face_position[3]))
                cv2.putText(image, person, (text_tl), cv2.FONT_HERSHEY_SIMPLEX, (face_position[2] - face_position[0])/140., (255, 0, 100), 1)

            # Display the resulting frame
            import random
            cv2.imwrite('./results/' + str(random.randint(0, 10000)) + '.png', image)
            cv2.imshow('Result', image)
            print('caculating in ', time.time() - start)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", help="location of input image")
    args = parser.parse_args()
    main()