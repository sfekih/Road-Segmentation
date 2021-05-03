from model import Model
from utils.metrics import *
from tensorflow import keras
from tensorflow.keras import datasets, layers, models,regularizers
from keras.layers import ReLU, LeakyReLU, BatchNormalization, Conv2D,\
     MaxPool2D, Dropout, Input, AveragePooling2D, Dense
import os


class Convolutional_neural_net (Model):
    """
    Implementation of a Convolutional Neural Network CNN using keras
    4 convolutional layers
    3 dense layers
    """
    def __init__ (self, input_shape=None, build_on_init=True, gpu=True,activation_Leaky=True):
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


    def set_input_shape (self, input_shape):
        """ 
        Input shape setter
        """
        self.input_shape = input_shape
        self.img_size=input_shape[0]
        self.n_channels=input_shape[2]

    
    def _apply_convolution(self, input, n_filters=16, kernel_size=3, activation_Leaky=True , alpha=0.1 , dropout_rate=0.25 ,padding_type='same',pool_size_dropout=(2,2),strides_shape=(2, 2)):
        """
        function to do one whole convolutional layer
        1) Apply convolution
        2) Apply Batch Normalization
        3) Apply activation function
        4) Apply maxpooling
        5) Apply Dropout
        """
        print(input.shape)
        #first convolutional layer
        conv1 =  Conv2D(filters=n_filters,
                        kernel_size= kernel_size,
                        kernel_initializer= 'he_normal',
                        strides=strides_shape,
                        padding='same')(input)
        
        #batch normalization
        norm = BatchNormalization()(conv1)

        #activation function
        if activation_Leaky:
            act= LeakyReLU(alpha)(norm)
        else: 
            act= ReLU(alpha)(norm)

        #maxpooling
        pool=MaxPool2D(pool_size=pool_size_dropout, padding=padding_type)(act)

        #dropout: written as a variable for clarity
        drop=Dropout(dropout_rate)(pool)
        
        return drop

    def build_model(self): 
        """
        Function made to build the CNN model
        We use four metrics : accuracy, F1 score, precision and recall
        """
        #Input layer  
        INPUT = Input((self.img_size, self.img_size , self.n_channels))

        
        #Apply four convolutional layers              
        layer1 = self._apply_convolution(input=INPUT,  kernel_size=5,  n_filters=64,   activation_Leaky=self.activation_Leaky)
        layer2 = self._apply_convolution(input=layer1, kernel_size=3,  n_filters=128,  activation_Leaky=self.activation_Leaky)
        layer3 = self._apply_convolution(input=layer2, kernel_size=3,  n_filters=256,  activation_Leaky=self.activation_Leaky)
        layer4 = self._apply_convolution(input=layer3, kernel_size=3,  n_filters=256,  activation_Leaky=self.activation_Leaky)
        
        #Flatten
        flat = keras.layers.Flatten()(layer4)
        
        #set alpha for the dense layers
        alpha=0.01

        #Apply three dense layers
        dense1 = Dense(units=128,kernel_regularizer=regularizers.l2(1e-6))(flat)

        #activation function
        if self.activation_Leaky:
            dense1= LeakyReLU(alpha)(dense1)
        else: 
            dense1= ReLU(alpha)(dense1)
        

        dense2 = Dense(units=64,kernel_regularizer=regularizers.l2(1e-6))(dense1)

        #activation function
        if self.activation_Leaky:
            dense2= LeakyReLU(alpha)(dense2)
        else: 
            dense2= ReLU(alpha)(dense2)
        
        dense2= Dropout(0.5)(dense2)

        dense3 = Dense(units=2,kernel_regularizer=regularizers.l2(1e-6),
                                    activation='softmax')(dense2)
        
        self.model = keras.Model(inputs=INPUT,outputs=dense3)
        print(self.model.summary())
        self.model.compile(optimizer='adam', metrics=['acc', f1_m, precision_m, recall_m],\
                            loss='binary_crossentropy')
        return self.model


    def predict(self, X): 
      return  self.model.predict(X) 
      






    
     
 
        
