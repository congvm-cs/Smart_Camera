""" This tool written by VMC helps you detect face from video and extract 
    face cropped image from every single frame based on facenet model.

    Requirement:
        Python 3.6
        Tensorflow

    Systax: 
        python face_detection_from_video.py --video_dir <input_video_location>
"""

import tensorflow as tf
import numpy as np
import os
import detect_face
import facenet
import argparse
import skvideo.io
import cv2

def to_rgb(img):
  w, h = img.shape
  ret = np.empty((w, h, 3), dtype=np.uint8)
  ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
  return ret


def get_video_location(video_dir):
    return os.path.split(video_dir)[0]    


def main():
    # mtcnn parameters
    minsize = 20                    # minimum size of face
    threshold = [ 0.4, 0.5, 0.6 ]   # three steps's threshold
    factor = 0.3                    # scale factor

    # restore mtcnn model
    print('Creating networks and loading parameters')
    gpu_memory_fraction=1.0
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, './model_check_point/')
    print('Creating networks completely!')

    # Get faces from video
    label = str(input("Enter image's name prefix: \n> "))
    
    # Loading video
    frames = skvideo.io.vreader(args.video_dir)
    frame_counter = 0
    face_index = 0

    print('-------------------------Processing---------------------------')
    # Processing on every single frame
    for frame in frames:
        if (frame_counter % 5) == 0:
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]    # number of faces
            print('Number of faces: {}'.format(nrof_faces))

            for face_position in bounding_boxes:    
                face_position = face_position.astype(int)
                # Get crop image from bounding box
                crop = frame[face_position[1]:face_position[3],face_position[0]:face_position[2], :]
                
                # Resize images
                crop = cv2.resize(crop, (160, 160))

                # Create crop image
                skvideo.io.vwrite(get_video_location(args.video_dir) + '/' + label + '-' + str(face_index) + '.png', crop)
                print(get_video_location(args.video_dir) + '/' + label + '-' + str(face_index) + '.png')
                face_index += 1
        frame_counter += 1      
    # When everything is done, release the capture
    print('\nFound: {} face(s)'.format(face_index))
    print('---------------------------Done-------------------------------')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("video_dir", help="location of input video.")
    args = parser.parse_args()
    main()
