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

class ISFace(object):
    def __init__(self):
        print('Loading pretrain model from disk...')
        self.facenet_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/20170512-110547.pb'
        self.mtcnn_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/model_check_point/'
        self.classification_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/classification_model.pkl'
        self.outlier_detector_model_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/outlier_model.pkl'
        self.embedded_data_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/embedded_data.pkl'

        self.sess = tf.Session()
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, self.mtcnn_model_dir)
        self.minsize = 20  # minimum size of face
        self.threshold = [0.4, 0.5, 0.6]  # three steps's threshold
        self.factor = 0.709  # scale factor
        self.frame_interval = 5
        self.image_size = 160
        self.input_image_size = 160
        self.batch_size = 40
        self.seed = 10
        self.test_size = 0.3

        self.load_model()

    def preprocess_cropped_image(self, cropped):
        cropped = facenet.flip(cropped, False)
        scaled = misc.imresize(cropped, (self.image_size, self.image_size), interp='bilinear')  ####
        scaled = cv2.resize(scaled, (self.input_image_size, self.input_image_size),
                            interpolation=cv2.INTER_CUBIC)

        scaled = facenet.prewhiten(scaled)
        scaled_reshape = scaled.reshape(-1, self.nput_image_size, self.input_image_size, 3)
        return scaled_reshape

    
    def embed_cropped_face():
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)?
            with self.sess.as_default():        
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                self.emb_array = sess.run(embeddings, feed_dict=feed_dict)


    def load_data_from_folder(self, data_dir):
        pass


    def predict(self):
        pass


    def fit(self, data_dir, estimator='svm', save_embeded_data=False, save_model = False):
        dataset = facenet.get_dataset(data_dir)        
        paths, labels = facenet.get_image_paths_and_labels(dataset)

        print('Number of classes: %d' % len(dataset))
        print('Number of images: %d' % len(paths))
        print('Number of labels: %d' % len(labels))
        print('Calculating features for images')
        nrof_images = len(paths)
        nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / self.batch_size))          
        emb_array = np.zeros((nrof_images, self.embedding_size))

        for i in range(nrof_batches_per_epoch):
            start_index = i*self.batch_size
            end_index = min((i+1)*self.batch_size, nrof_images)
            paths_batch = paths[start_index:end_index]
            images = facenet.load_data(paths_batch, False, False, self.image_size)     
            feed_dict = {self.images_placeholder:images, self.phase_train_placeholder:False }
            emb_array[start_index:end_index,:] = self.sess.run(self.embeddings, 
                                                          feed_dict=feed_dict)
            print("Epoch #{}. Embedded: {}/{}".format(i, end_index, nrof_images))
    
        if save_embeded_data == True:
            with open(self.embedded_data_dir, 'wb') as pickle_file:
                pickle.dump(emb_array, pickle_file)

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(emb_array, 
                                                            labels, 
                                                            random_state = self.seed, 
                                                            test_size=self.test_size)

        if estimator == 'svm':
            model = SVC(kernel='linear', probability=True, C=12, gamma='auto')
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            from sklearn.metrics import accuracy_score
            test_score = accuracy_score(y_test, y_test_pred)
            train_score = accuracy_score(y_train, y_train_pred)

            print("Train accuracy: {}".format(train_score))
            print("Test accuracy: {}".format(test_score))

        if save_model == True:
            with open(self.classification_model_dir, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)


    def load_model(self):
        with tf.Graph().as_default():
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)?
            with self.sess.as_default():
                print('Loading Facenet Model')
                facenet.load_model(self.facenet_model_dir)
                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

        print('Loading classification model...')
        classifier_filename_exp = os.path.expanduser(self.classification_model_dir)
        with open(classifier_filename_exp, 'rb') as infile:
            (self.classification_model, self.class_names) = pickle.load(infile)
        print('load classifier file-> %s' % classifier_filename_exp)

        print('Loading outlier detector model')
        classifier_filename_exp = os.path.expanduser(self.outlier_detector_model_dir)
        with open(classifier_filename_exp, 'rb') as infile:
            self.outlier_detector_model = pickle.load(infile)
        print('load classifier file-> %s' % classifier_filename_exp)
        print('Done.')


    def face_localize(self, frame):
        if frame.ndim == 2:
            frame = facenet.to_rgb(frame)
        bounding_boxes, _ = detect_face.detect_face(frame, self.minsize, self.pnet, self.rnet, self.onet, self.threshold, self.factor)
        print('Nof.detected face(s): {}'.format(nrof_faces))
        return bounding_boxes


    def detect_from_image(self, image_dir):
        pass


    def realtime_detection(self):
        pass