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

def main():
    #=======================================================================================#
    print('Loading load from disk...')
    facenet_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/20170512-110547.pb'
    mtcnn_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/'
    classification_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model.pkl'
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
            image = cv2.imread(args.image_name, 1)
            
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            RGB_img = cv2.resize(RGB_img, (0,0), fx=0.5, fy=0.5)

            frame = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

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
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                    pred = outlier_detector_model.predict(emb_array)

                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20

                    # Save image
                    saved_face = cv2.resize(RGB_img[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :], (image_size, image_size))
                    skvideo.io.vwrite("./datasets/Faces/mat_" + str(i) + '.png', saved_face)

                    if pred == -1:
                        cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)

                    else:
                        predictions = classification_model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                        #plot result idx under box
                        proba = float(np.max(predictions))
                        str_proba = str("%.2f" % proba)
                        
                        for H_i in class_names:
                            if class_names[best_class_indices[0]] == H_i:
                                result_names = class_names[best_class_indices[0]]
                                cv2.putText(frame, result_names, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                                cv2.putText(frame, str_proba, (text_x, text_y + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)

                
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