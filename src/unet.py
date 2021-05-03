from model import Model
from utils.metrics import *
import numpy as np
from tensorflow import keras
from keras.optimizers import Adam
from tensorflow.keras import datasets, layers, models,regularizers
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Conv2D,\
     MaxPooling2D, Dropout, Input, AveragePooling2D, Dense, Conv2DTranspose, concatenate




class Unet (Model):

    def __init__ (self, input_shape=None, build_on_init=True, gpu = True ,activation_Leaky=True):
        """
        Initialize attributes
        """

        super().__init__(None, None, False, gpu)
        self.set_input_shape((None, None, None))
        self.activation_Leaky= activation_Leaky
        

        if (build_on_init):
            if (input_shape is None):  
                raise ValueError("input_shape needs to be specified if you want to build on initialization")
            else: 
                self.set_input_shape(input_shape)
                self.build_model()
        return 


    def set_input_shape (self, input_shape):
        """ 
        Input shape setter
        """
        self.input_shape = input_shape
        self.img_size=input_shape[0]
        self.n_channels=input_shape[2]
        return 

    

    def _conv(self, input_img, n_filters, kernel_size, activation_Leaky , alpha ):
        """
         Perform two convolutions each followed by a batch normalization and an activation function
        """

        #first convolutional layer
        conv1 =  Conv2D(filters=n_filters,
                        kernel_size= kernel_size,
                        kernel_initializer= 'he_normal',
                        padding='same')(input_img)
        
        #batch normalization
        conv1 = BatchNormalization()(conv1)

        #activation function
        if activation_Leaky:
            conv1= LeakyReLU(alpha)(conv1)
        else: 
            conv1= ReLU(alpha)(conv1)


        #second convolutional layer
        conv2= Conv2D(filters=n_filters, 
                        kernel_size= kernel_size, 
                        kernel_initializer=  'he_normal',
                        padding='same')(conv1)
        
        #batch normalization
        conv2 = BatchNormalization()(conv2)

        #activation function
        if activation_Leaky:
            conv2= LeakyReLU(alpha)(conv2)
        else :
            conv2= ReLU(alpha)(conv2)

        return conv2

    
    def down_sample(self, conv , dropout_val ):
        """
          down sampling setp consisting of a maxpooling followed by a potential dropout
        """

        pool = MaxPooling2D(pool_size=(2, 2) ,strides=None, padding='same', data_format='channels_last')(conv)

        if dropout_val != None: 
            pool = Dropout(dropout_val)(pool)

        return pool 

    def up_sample(self, conv1, conv2, n_filters, kernel_size ): 
        """
          up sampling step consisting of a transposed convolution that is then concatenated with the  corresponding  cropped
          feature  map  from the  contracting  path
        """

        up =Conv2DTranspose(n_filters, kernel_size= kernel_size, strides=(2, 2), padding='same') (conv1)

        merge = concatenate([conv2,up], axis = 3)

        return up, merge

 

    def build_model(self, n_filters= 16, kernel_size= 3, dropout_val=0.5 , lr=0, alpha=0.3):
        """
          build the unet model
          We use four metrics : accuracy, F1 score, precision and recall
        """
        input_imgs = Input((self.img_size, self.img_size , self.n_channels))

        
        #Encoder
        #1
        conv1= self._conv(input_imgs, n_filters*1, kernel_size, self.activation_Leaky, alpha )
        down1 = self.down_sample(conv1 , dropout_val)

        #2
        conv2= self._conv(down1, n_filters*2, kernel_size, self.activation_Leaky, alpha )
        down2 = self.down_sample(conv2 , dropout_val)

        #3
        conv3= self._conv(down2, n_filters*4,  kernel_size, self.activation_Leaky, alpha  )
        down3 = self.down_sample(conv3 , dropout_val)

        #4
        conv4= self._conv(down3, n_filters*8,  kernel_size, self.activation_Leaky , alpha )
        down4 = self.down_sample(conv4 , dropout_val)

        #5
        conv5= self._conv(down4, n_filters*16,  kernel_size, self.activation_Leaky, alpha  )
        drop5 = Dropout(dropout_val)(conv5)
        



        #Decoder 

        #6 
        up6, merge6= self.up_sample(drop5, conv4,  n_filters*8 , kernel_size) 
        conv6 = self._conv(merge6, n_filters*8 , kernel_size, self.activation_Leaky, alpha)

  
        #7
        up7, merge7= self.up_sample(conv6, conv3,  n_filters*4 , kernel_size )
        conv7 = self._conv(merge7, n_filters*4 , kernel_size, self.activation_Leaky, alpha )

        #8
        up8, merge8= self.up_sample(conv7, conv2,  n_filters*2 ,kernel_size )
        conv8 = self._conv(merge8,n_filters*2 , kernel_size, self.activation_Leaky, alpha )

        #9
        up9, merge9= self.up_sample(conv8, conv1,  n_filters*1 ,kernel_size )
        conv9 = self._conv(merge9, n_filters*1 , kernel_size, self.activation_Leaky, alpha )

        

        # 1 x 1 convolution
        output = Conv2D(1, 1, activation = 'sigmoid')(conv9)


          
        self.model = keras.Model(inputs = [input_imgs], outputs = [output])

        self.model.compile(optimizer = Adam(), loss='binary_crossentropy', metrics=['acc',f1_m, precision_m, recall_m])
        
        print(self.model.summary())

        return self.model



    def predict(self, imgs, padd_608, with_avg=True):
            
        """
        make predictions depending on the given parameters
        """
        if(padd_608):
          #if the train images are of the same size as the test images (608*608*3) we just predict
          predictions= self.model.predict(imgs)

        else: 
          if(imgs.shape[1]==400):
            #when we split the train set and evaluate the model on images of 400*400 pixels
            predictions= self.model.predict(imgs) 

          elif(with_avg==False):
            #when we make predictions on tets images of 608*608 pixels 
            #and take one of the predictions on the overlapping parts
            predictions=[]
            for X in imgs:
              
              img1 = np.expand_dims(X[:400, :400], axis=0)
              img2 = np.expand_dims(X[:400, -400:], axis=0)
              img3 = np.expand_dims(X[-400:, :400], axis=0)
              img4 = np.expand_dims(X[-400:, -400:], axis=0)
              
              prediction = np.zeros((X.shape[0], X.shape[1], 1))
              
              prediction[:400, :400] = np.squeeze(self.model.predict(img1), axis=0)
              prediction[:400, -400:] = np.squeeze(self.model.predict(img2), axis=0)
              prediction[-400:, :400] = np.squeeze(self.model.predict(img3), axis=0)
              prediction[-400:, -400:] = np.squeeze(self.model.predict(img4), axis=0)
              predictions.append(prediction)

          else: 
            #when we make predictions on tets images of 608*608 pixels 
            #and take the avergae of the predictions on the overlapping parts
            predictions=[]
            for X in imgs:
                dist=200
                img1 = np.expand_dims(X[:400, :400], axis=0)
                img2 = np.expand_dims(X[:400, -400:], axis=0)
                img3 = np.expand_dims(X[-400:, :400], axis=0)
                img4 = np.expand_dims(X[-400:, -400:], axis=0)
                
                prediction = np.zeros((X.shape[0], X.shape[1], 1))
                
                prediction[:400, :400] += np.squeeze(self.model.predict(img1), axis=0)
                prediction[:400, -400:] += np.squeeze(self.model.predict(img2), axis=0)
                prediction[-400:, :400] += np.squeeze(self.model.predict(img3), axis=0)
                prediction[-400:, -400:] += np.squeeze(self.model.predict(img4), axis=0)

                #1 and 3
                prediction[dist: 600-dist, :dist] = prediction[dist: 600-dist, :dist]/2.0 

                #2 and 4
                prediction[dist: 600-dist, 600-dist:] = prediction[dist: 600-dist, 600-dist:] /2.0

                #1 and 2
                prediction[:dist, dist:600-dist] = prediction[:dist, dist:600-dist]/2.0 

                #3 and 4
                prediction[600-dist:, dist:600-dist] = prediction[600-dist:, dist:600-dist]/2.0 

                #1 and 2 and 3 and 4
                prediction[dist:600-dist, dist:600-dist] = prediction[dist:600-dist, dist:600-dist]/4.0

                predictions.append(prediction)


        return np.asarray(predictions)




    



 



            


     
 
        
