import dill
import pickle
import cv2
#This would be able to load the previously trained decoder and encoder

import numpy as np
a = np.load("/home/xiaomin/Downloads/IFIG_DATA_VAE_1000_10000/0.npy")


encoded = a[1]

print(encoded)
print('------------------------------')

with open("/home/xiaomin/Downloads/IFIG_DATA_VAE_300_10000/get_img.txt", "rb") as fp:
    c = pickle.load(fp)
func_get_img = dill.loads(c)

img_1_reconstruct = func_get_img(encoded).transpose()
print(img_1_reconstruct)
#dataset[i, :] = unormalize_image(img)
# img_1 = img_1.reshape(3, imsize, imsize).transpose()
cv2.imshow('test', img_1_reconstruct)
cv2.waitKey(0)



import dill
import pickle
get_encoded = dill.dumps(vae_environment._get_encoded)
with open("/home/xiaomin/Downloads/IFIG_DATA_VAE_300_10000/get_encoded.txt", "wb") as fp:
    pickle.dump(get_encoded, fp)
with open("/home/xiaomin/Downloads/IFIG_DATA_VAE_300_10000/get_encoded.txt", "rb") as fp:
    b = pickle.load(fp)
func_get_encoded = dill.loads(b)
encoded = func_get_encoded(obs['image_observation'])
print(encoded)
print('------------------------------')
get_img = dill.dumps(vae_environment._get_img)
with open("/home/xiaomin/Downloads/IFIG_DATA_VAE_300_10000/get_img.txt", "wb") as fp:
    pickle.dump(get_img, fp)
with open("/home/xiaomin/Downloads/IFIG_DATA_VAE_300_10000/get_img.txt", "rb") as fp:
    c = pickle.load(fp)
func_get_img = dill.loads(c)

img_1_reconstruct = func_get_img(encoded).transpose()
print(img_1_reconstruct)
#dataset[i, :] = unormalize_image(img)
# img_1 = img_1.reshape(3, imsize, imsize).transpose()
cv2.imshow('test', img_1_reconstruct)
cv2.waitKey(0)
