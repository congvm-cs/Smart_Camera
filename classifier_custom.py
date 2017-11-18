""" Written by VMC
    This tool written for train your own dataset
"""
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import math
import pickle
from sklearn.svm import SVC
from sklearn.utils import shuffle

reload_data = False
if reload_data:
    print('Mode: reload data.')
else:
    print('Mode: use saved data.')

def write_embedding_data(X, y, file_name):
    with open(file_name, 'w+') as file:
        for i in range(X.shape[0]):
            for item in X[i, :]:
                file.write(str(item) + ' ')
            file.write(y[i] + '\n')


def main():    
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset(args.data_dir)        
            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            print('Number of labels: %d' % len(labels))

            # Load the model
            print('Loading feature extraction model')
            model_path = './model_check_point/20170512-110547.pb'
            facenet.load_model(model_path)                      

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            
            # Run forward pass to calculate embeddings
            batch_size = 40
            image_size = 160

            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / batch_size))          
            emb_array = np.zeros((nrof_images, embedding_size))
            
            if reload_data:
                for i in range(nrof_batches_per_epoch):
                    start_index = i*batch_size
                    end_index = min((i+1)*batch_size, nrof_images)
                    paths_batch = paths[start_index:end_index]
                    images = facenet.load_data(paths_batch, False, False, image_size)     
                    feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                    emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
                    print("Epoch #{}. Embedded: {}/{}".format(i, end_index, nrof_images))
            
                # save data on disk
                with open("emb_array.pkl", 'wb') as pickle_file:
                    pickle.dump(emb_array, pickle_file)
            else:
                # load data from disk
                with open("emb_array.pkl", 'rb') as pickle_file:
                    emb_array = pickle.load(pickle_file)

            # Train classifier
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(emb_array, labels, random_state = 10, test_size=0.3)
            
            print('Training classifier')
            model = SVC(kernel='linear', probability=True, C=12, gamma='auto')
            model.fit(X_train, y_train)

            # # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]
            classifier_filename_exp = os.path.expanduser(args.classifier_filename)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            from sklearn.metrics import accuracy_score
            test_score = accuracy_score(y_test, y_test_pred)
            train_score = accuracy_score(y_train, y_train_pred)

            print("Train accuracy: {}".format(train_score))
            print("Test accuracy: {}".format(test_score))

            # Write into file
            y = []
            for i in labels:  
                y.append(class_names[i])

            # write_embedding_data(emb_array, y, 'embedding_Data')

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, action='store', dest='data_dir',
                    help='data directory')
    parser.add_argument('--cls-name', dest='classifier_filename',
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    args = parser.parse_args()
    main()