from isface import ISFace

data_dir = '/media/vmc/Data/VMC/Workspace/Smart-Camera/datasets/Faces'
face_dec = ISFace()
face_dec.fit(data_dir, estimator='svm', save_embeded_data=True, save_model = True)
