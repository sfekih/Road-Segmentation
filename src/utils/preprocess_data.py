# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator
from skimage import io

import tensorflow as tf
from tensorflow import keras

from sklearn.feature_extraction import image

from sklearn.model_selection import train_test_split

from utils.data_augmentation import *
from utils.load_helpers import *

def load_images(image_dir, gt_dir):
    """
    Load training set images
    """
    files = os.listdir(image_dir)
    imgs = np.asarray([load_image(os.path.join(image_dir, files[i])) for i in range(len(files))])
    gt_imgs = np.asarray([load_image(os.path.join(gt_dir, files[i])) for i in range(len(files))]) 

    return imgs, gt_imgs


def load_images_test(dir, Unet):
    """
  Load test set images the return them in the needed shape
  The needed shape is different depending on the model used
  If we work with unet, we keep the test image as one (the whole image is fed to Unet)
  If we work with CNN, we crop the image into other images of size 16*16 and return
    both the original images and the patches derived
  """
  # Loaded a set of images from directories
  files_=[os.path.join(dir,'test_'+str(i),'test_'+str(i)+'.png') for i in range (1,51)]
  imgs = np.asarray([load_image(files_[i]) for i in range(len(files_))])
  

  n = len(imgs)

  if(Unet==False):
      # Extract patches from input images
      patch_size = 16 # each patch is 16*16 pixels

      img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
      img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

      #pad the test images 
      #imgs_pad = np.pad(img_patches, ((152, 400), (152, 400)), 'reflect')
      #print(img_pad.shape)
      return img_patches, imgs
  else: 
    return  imgs 


  


def data_augment( imgs, gt_imgs,image_dir, gt_dir, max_iters ):
    """
    augment our training dataset
    """
    
    augment_dataset2(imgs,gt_imgs,image_dir,gt_dir,max_iters)

    # Loaded a set of images
    files_aug = os.listdir(image_dir)
    imgs_aug = np.asarray([load_image(image_dir + files_aug[i]) for i in range(len(files_aug))])

    files_gt = os.listdir(gt_dir)
    gt_imgs_aug =np.asarray([ (load_image(gt_dir + files_aug[i]) if i<100  else load_image(gt_dir + files_gt[i]) ) for i in range(len(files_aug))])
    
    return imgs_aug, gt_imgs_aug

def get_patches(imgs, gt_imgs):
    n = len(imgs)

    # Extract patches from input images
    patch_size = 16 # each patch is 16*16 pixels

    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    
    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))]) 

    
    return img_patches, gt_patches

def compute_features(img_patches, gt_patches , foreground_threshold):
    """
  Compute features for each image patch
  """  

    X = np.asarray([ img_patches[i] for i in range(len(img_patches))])
    Y = np.asarray([value_to_class_NN(gt_patches[i],foreground_threshold) for i in range(len(gt_patches))])
    
    return X, Y

def split(X, Y,test_ratio=0.2 ,  random_state=0):
  #split into train and test sets
  return train_test_split(X,Y, test_size=test_ratio, random_state=random_state )

def preprocess(root_dir, max_iters=20, random_state=0, test_ratio=0.2 , foreground_threshold = 0.25 , Unet=False, padd_608=False):
    """
  main function to perform data augmentation
  INPUTS:
  1) root_dir : root directory of training images
  2) max_iters : the more this parameter is big, the more images will be created
  3) test_ratio : ratio of images in the test set, 
  set test_ratio to 0 if you want to feed the model with the wole training images set
  4) foreground_threshold = 0.25 is percentage of pixels > 1 required to assign a foreground label to a patch
  5) Unet : boolean : whether you are working with a Unet or not
  6) padd_608 : boolean : whether you want to feed your Unet, with images of shape 608*608 or not

  OUTPUTS:
  training set and test set
  """
  #foreground_threshold = 0.25 is percentage of pixels > 1 required to assign a foreground label to a patch

    #get the images directory
    image_dir = os.path.join(root_dir, "images/")

    #get the grounftruth directory
    gt_dir =  os.path.join(root_dir, "groundtruth/")

    #Load the images and the corresponding grounthruths
    imgs, gt_imgs= load_images(image_dir, gt_dir)

      
    #perform data augmentation
    imgs_aug, gt_imgs_aug =data_augment(imgs, gt_imgs, image_dir, gt_dir ,max_iters )
        
    


    if( Unet):

      print("unet")

      X= imgs_aug

      Y= np.expand_dims( gt_imgs_aug, axis=3)

      if(padd_608):
        #padd the train images to get images of 608*608 pixels
        X=np.pad(X,[(0,0),(104,104),(104,104),(0,0)],mode='symmetric')
        Y=np.pad(Y,[(0,0),(104,104),(104,104),(0,0)],mode='symmetric')
      


      
    else:
      #divide images into patches
      img_patches, gt_patches=get_patches(imgs_aug, gt_imgs_aug)

      # Compute features for each image patch
      X, Y =compute_features(img_patches, gt_patches , foreground_threshold)
      

    if(test_ratio>0):
      #split the data into train and test sets
      X_train,X_test,Y_train,Y_test=train_test_split(X,Y, test_size=test_ratio, random_state=random_state)
    else:
      X_train,X_test,Y_train,Y_test= X, None, Y , None
    return X_train,X_test,Y_train,Y_test
