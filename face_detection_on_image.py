
# coding: utf-8

# This script obtaining frames from camera,using mtcnn detecting faces,croping 
# and embedding faces with pre-trained facenet and finally face recogition with 
# pre-trained classifier.
import tensorflow as tf
import numpy as np
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
import argparse

def to_rgb(img):
    """Convert from gray 1 channel to gray 3 channels
    """
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret

def main():
    #face detection parameters
    minsize = 20 # minimum size of face
    threshold = [ 0.5, 0.6, 0.7 ]  # three steps's threshold
    factor = 0.5 # scale factor

    #restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction=0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default(): 
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
    print('Creating networks completely')

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

    for face_position in bounding_boxes:
        face_position=face_position.astype(int)
        
        cv2.rectangle(image, (face_position[0], 
                        face_position[1]), 
                        (face_position[2], face_position[3]), 
                        (0, 255, 0), 2)
        
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

    # Display the resulting frame
    cv2.imshow('Result', image)
    print('caculating in ', time.time() - start)
    cv2.imwrite('./results/' + str(random.randint(0, 10000)) + '.png', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("img_dir", help="location of input image")
    args = parser.parse_args()
    main()