from utils.load_helpers import plot_img_pred_and_overlay, img_crop
from utils.preprocess_data import load_images_test

import os
import numpy as np

def patch_to_label(patch):
    """
    assign a label to a patch
    """
    foreground_threshold=0.25
    df = np.mean(patch)
    if df > foreground_threshold:
        return 1
    else:
        return 0


def get_prediction_csv (model,TEST_PATH, submission_filename, Unet=True, padd_608=False, with_avg=True,plot_images=True,number_images_to_plot=1):
    """
    function to get the file submission for Aicrowd
    INPUTS:
    1) model : the model we use
    2) TEST_PATH : path for the test images
    3) Unet : boolean : whether our model is a Unet or not
    4) padd_608 : boolean : whether the input image was padded to augment its dimentions to 608*608 or not
    5) with_avg : boolean : applied to 400*400 size images : 
    if with_avg is set to be True : we take to average of the overlapping prediction
    if with_avg is set to be False : we also keep one prediction (the last one)
    7) plot_images : if you want to visualize the predictions
    8) number of images you want to visualize

    OUTPUTS:
    csv file containing the predictions
    
    Since Unet and CNN have different input sizes and work with different methods (Unet work with whole image wheras CNN works with patches),
    We make different output methods for eaach model
    ***If we work with Unet, we first begin by doing the prediction(whether with 4 images of size 400x400 or with one image of size 608),
    then we crop the output into patches of size 16x16  for our predictions
    ***If we work with CNN, we import cropped images then we do our predictions. 
    If the user wants to have plots, we expand the dimensions of the patches to have 608x608 images ready for plots

    Finally, we prepare the submission file 
    """
    foreground_threshold=0.25
    patch_size=16
    
    if Unet:
        #load the test images
        imgs_test=load_images_test(os.path.join(TEST_PATH),Unet=Unet)

        #get the predictions using the model
        predictions_test=model.predict(imgs_test, padd_608, with_avg)[:,:,:,0]

        #crop each prediction into the format needed for submission
        #First get the shapes and variables we need
        number_patches_per_dim=predictions_test.shape[1]//patch_size
        number_images_test=predictions_test.shape[0]
        #Then crop and put the images in the wanted shape
        predictions_cropped=[img_crop(predictions_test[i], patch_size, patch_size) for i in range(number_images_test)]
        predictions_cropped = np.asarray([predictions_cropped[i][j] for i in range(len(predictions_cropped)) for j in range(len(predictions_cropped[i]))])
        predictions_cropped = np.asarray([patch_to_label(predictions_cropped[i]) for i in range(len(predictions_cropped))]).\
                                reshape(number_images_test,number_patches_per_dim,number_patches_per_dim)
                                    

    else : #If CNN
        #load test images and the derived patches
        patches_test,imgs_test = load_images_test(os.path.join(TEST_PATH),Unet=Unet)
        test_image_shape=imgs_test.shape[:-1]
        number_images_test=test_image_shape[0]

        #get the predictions using the model, get only second column : the one that contains the predictions
        predictions_test=model.predict(patches_test)[:,1]

        # make our predictions binary (0 or 1) and reshape the vector in order to have a good form for submission
        number_patches_per_dim=int(np.sqrt(len(predictions_test)//number_images_test))
        predictions_cropped=np.asarray([1 if predictions_test[i]>foreground_threshold else 0 for i in range (len(predictions_test)) ]).\
                        reshape(number_images_test,number_patches_per_dim,number_patches_per_dim)
        if plot_images:
            #set a new vector predictions_test and prepare it for future plots
            # Only do this work for images we will need to plot
            predictions_test=np.empty(test_image_shape)
            for im in range(number_images_to_plot):
                for w in range(0,test_image_shape[2],patch_size):
                    for l in range(0,test_image_shape[1],patch_size):
                        predictions_test[im,l:l+patch_size,w:w+patch_size]=predictions_cropped[im,w//patch_size,l//patch_size]
                
    if plot_images:
        for i in range(number_images_to_plot):
            plot_img_pred_and_overlay(imgs_test[i],predictions_test[i,:,:],PIXEL_DEPTH=255)
        
    imgs_cropped_shape=predictions_cropped.shape

    #save the predictions in a submission csv file
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for im in range(imgs_cropped_shape[0]):
            for w in range(imgs_cropped_shape[2]):
                for l in range(imgs_cropped_shape[1]):
                    f.writelines('{:03d}_{}_{},{}\n'.format(im+1,w*16,l*16,predictions_cropped[im,w,l]))
