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

print('Creating networks and loading parameters')
with tf.Graph().as_default():
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)?
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/')

        minsize = 20  # minimum size of face
        threshold = [0.5, 0.6, 0.6]  # three steps's threshold
        factor = 0.709  # scale factor
        frame_interval = 3
        image_size = 160
        input_image_size = 160

        print('Loading feature extraction model')
        modeldir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/20170512-110547.pb'
        facenet.load_model(modeldir)

        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]

        classifier_filename = '/media/vmc/Data/VMC/Workspace/Smart-Camera/outlier_model.pkl'
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            model = pickle.load(infile)
            print('load classifier file-> %s' % classifier_filename_exp)

        video_capture = cv2.VideoCapture(0)
        c = 0

        print('Start Recognition!')
        prevTime = 0
        while True:
            ret, frame = video_capture.read()

            frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

            curTime = time.time()    # calc fps
            timeF = frame_interval

            if (c % timeF == 0):
                find_results = []

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
                        scaled_reshape = scaled.reshape(-1,input_image_size,input_image_size,3)
                        feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                        # Prediction
                        predictions = model.predict(emb_array)

                        #plot result idx under box
                        cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face
                        text_x = bb[i][0]
                        text_y = bb[i][3] + 20
                        if (predictions == -1):
                            cv2.putText(frame, "Unknown", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            cv2.putText(frame, "Known", (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (0, 0, 255), thickness=1, lineType=2)
                else:
                    print('Unable to align')

            sec = curTime - prevTime
            prevTime = curTime
            fps = 1 / (sec)
            strs = 'FPS: %2.3f' % fps
            text_fps_x = len(frame[0]) - 150
            text_fps_y = 20
            cv2.putText(frame, strs, (text_fps_x, text_fps_y),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
            
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
