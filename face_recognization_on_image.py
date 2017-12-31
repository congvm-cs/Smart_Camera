import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import time
import math
import pickle
from sklearn.svm import SVC
import skvideo.io
import random

def draw_rectangle(img, p1, p2, p3, p4, str_name, str_accuracy, color):
    offset = int((p3 - p1)/4)
    thickness_heavy_line = 3
    thickness_slim_line = 1
    text_x = p1
    text_y = p4 + 20
    # Left Top (p1, p2)
    cv2.line(img, (p1, p2), (p1, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p2), (p1 + offset, p2 ), color, thickness_heavy_line)
    
    # Left Bottom (p1, p4)
    cv2.line(img, (p1, p4), (p1, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p1, p4), (p1 + offset, p4 ), color, thickness_heavy_line)

    # Right Top (p3, p2)
    cv2.line(img, (p3, p2), (p3, p2 + offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p2), (p3 - offset, p2), color, thickness_heavy_line)

    # Right Bottom (p3, p4)
    cv2.line(img, (p3, p4), (p3, p4 - offset), color, thickness_heavy_line)
    cv2.line(img, (p3, p4), (p3 - offset, p4 ), color, thickness_heavy_line)
    
    cv2.line(img, (p1, p2), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p1, p2), (p3, p2), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p1, p4), color, thickness_slim_line)
    cv2.line(img, (p3, p4), (p3, p2), color, thickness_slim_line)

    cv2.putText(img, str_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, color, thickness=1, lineType=2)
    
    if not str_name == 'Unknown':
        cv2.putText(img, str_accuracy, (text_x, text_y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1, color, thickness=1, lineType=2)
    return img


def main():
    #=======================================================================================#
    print('Loading load from disk...')
    facenet_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/20170512-110547.pb'
    mtcnn_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/'
    classification_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/classification_model.pkl'
    outlier_detector_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/outlier_model.pkl'
    
    #=======================================================================================#

    with tf.Graph().as_default():
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)?
        sess = tf.Session()
        with sess.as_default():
            
            #================================================================================#
            print('Creating networks and loading parameters...')
            pnet, rnet, onet = detect_face.create_mtcnn(sess, mtcnn_model_dir)
            minsize = 20  # minimum size of face
            threshold = [0.5, 0.6, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            frame_interval = 3
            image_size = 160
            input_image_size = 160

            #================================================================================#
            print('Loading feature extraction model')
            facenet.load_model(facenet_model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            print('Done.')
            
            
            #================================================================================#
            print('Loading classification model...')
            classifier_filename_exp = os.path.expanduser(classification_model_dir)
            with open(classifier_filename_exp, 'rb') as infile:
                (classification_model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)

            print('Loading outlier detector model')
            classifier_filename_exp = os.path.expanduser(outlier_detector_model_dir)
            with open(classifier_filename_exp, 'rb') as infile:
                outlier_detector_model = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)


            #================================================================================#
            print('Start Recognition!')
            frame = cv2.imread(args.image_name, 1)
            RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)

            frame = frame[:, :, 0:3]

            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        continue

                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    cropped = facenet.flip(cropped, False)
                    scaled = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                    scaled = cv2.resize(scaled, (input_image_size,input_image_size),
                                        interpolation=cv2.INTER_CUBIC)

                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1,input_image_size,input_image_size, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                    #=============================================================================================#
                    pred = outlier_detector_model.predict(emb_array)
                   
                    if pred == -1:
                        frame = draw_rectangle(frame, bb[i][0], bb[i][1], bb[i][2], bb[i][3], 'Unknown', None, (0, 0, 255))
                    else:
                        predictions = classification_model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        # plot result idx under box
                        proba = float(np.max(predictions))
                        str_proba = str("%.2f" % proba)
                        
                        for H_i in class_names:
                            if class_names[best_class_indices[0]] == H_i:
                                result_names = class_names[best_class_indices[0]]

                                frame = draw_rectangle(frame, bb[i][0], bb[i][1], bb[i][2], bb[i][3], result_names, str_proba, (0, 255, 0))

                                

                                # skvideo.io.vwrite("./datasets/Faces/" + result_names + "_" + str(random.randint(0, 999999)) + '.png', saved_face)
                cv2.imshow('Image', frame)
                cv2.waitKey(0)       

            else:
                print('Unable to align')

if __name__ == '__main__':  
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-name', type=str, action='store', dest='image_name',
                    help='image directory')
    args = parser.parse_args()
    main()