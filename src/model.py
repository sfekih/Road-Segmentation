from abc import ABC, abstractmethod
import os
import pickle
import numpy as np 
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import models
from utils.metrics import *
import torch 


class Model(ABC):

    def __init__ (self, model, history, loaded_trained, gpu):
        self.model = model
        self.history = history
        self.loaded_trained = loaded_trained
        if gpu: 
            if torch.cuda.is_available():
                torch.device("cuda")
                print("using GPU !")
            else: 
                raise ValueError("you don't have cuda please set it up and try again")
        else: 
            torch.device("cpu")


    @abstractmethod
    def build_model(self):
        """
        abstract method : used to build the model : implement the Neural Network architecture
        """
        pass
    

    def serialize(self, name="", path="trained_models"): 
        """
        serialize our model in order to store it and be able to use it again
        """
        if not os.path.isdir(os.path.join("..",path)): 
            raise ValueError("path is incorrect")
        else:
          
            #save the model
            print("saving the model")
            self.model.save(os.path.join("..",path, name))
            
            #save the history
            with open(os.path.join("..", "history", name), 'wb') as file_pi:
                  pickle.dump(self.history.history, file_pi)

    
    def load_serialized(self, path_model, path_history = None):
        """
        load an already serialized model
        """
        #load the trained model
        self.model = models.load_model(path_model, custom_objects={ "f1_m":f1_m, "precision_m": precision_m, "recall_m":recall_m }) 
        self.loaded_trained = True
        
        #load the history when a path for it is provided
        if (path_history is not None):
            with open(path_history, 'rb') as file_pi:
                self.history = pickle.load(history.history, file_pi)
            return True
        return False

    def predict(self, X):
        """
        after fitting our model, we use it to predict the output of the testing set
        """
        if (not self.loaded_trained): 
            raise ValueError("Train or load a model before prediction")
        pass 


    def plot_history (self, dir_plots):
        """
        plot the accuracy, the F1 score, the precision and the recall as a function of the number of epochs
        """
        if not self.loaded_trained: 
            raise ValueError("Train or load a model beforehand")
            if self.history is None: 
                raise ValueError("History unavailable")
        # list all data in history
        print(self.history.history.keys())
        # summarize history for accuracy
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].plot(self.history.history['acc'])
        axs[0, 0].set_title('model accuracy')
        axs[0, 0].set_ylabel('accuracy')
        axs[0, 0].set_xlabel('epoch')  
        # summarize history for f1 score
        axs[0, 1].plot(self.history.history['f1_m'])
        axs[0, 1].set_title('model f1 score')
        axs[0, 1].set_ylabel('f1 score')
        axs[0, 1].set_xlabel('epoch')
        # summarize history for precision
        axs[1, 0].plot(self.history.history['precision_m'])
        axs[1, 0].set_title('model precision')
        axs[1, 0].set_ylabel('precision')
        axs[1, 0].set_xlabel('epoch')
        # summarize history for recall 
        axs[1, 1].plot(self.history.history['recall_m'])
        axs[1, 1].set_title('model recall')
        axs[1, 1].set_ylabel('recall')
        axs[1, 1].set_xlabel('epoch')

        plt.tight_layout()

        #save the plot
        plt.savefig(dir_plots)

        plt.show()

    def evaluate(self, X, y):
        """
        Evaluate the model:
        We have the trained model and we want to evaluate its performace on some test data 
        X : test input
        Y : test output
        """
        return self.model.evaluate(X,y)

    def _callbacks(self):
        """
        function to get early stopper and lr_reducer in our model
        """

        #Stop training when f1_m metric has stopped improving for 20 epochs
        earlystopper = EarlyStopping(monitor =f1_m, 
                                    mode='max', 
                                    patience = 5,
                                    verbose = 1,
                                    restore_best_weights = True)

        #Reduce learning rate when loss has stopped improving for 3 epochs
        lr_reducer = ReduceLROnPlateau(monitor='loss',
                                    mode='min',
                                    factor=0.9,
                                    patience=2,
                                    min_delta= 0.001, 
                                    min_lr=0.00001,
                                    verbose=1)

        return [earlystopper, lr_reducer]

    def fit (self, X, Y,  dir_plots, epochs=600, batch_size=16, class_weights = None, plots=True ) : 
        """
        We fit our model using the training data (X train and Y train)
        We can choose the number of epochs to use as well as the batch size
        We can also choosee to do plots or not to our history
        """
        assert(X.shape[1:] == self.input_shape)

        self.history=None
        try: 
          self.history = self.model.fit(X, Y, epochs=epochs, batch_size=batch_size, \
                                    use_multiprocessing=True, workers = os.cpu_count(),callbacks=self._callbacks())
        except KeyboardInterrupt:
            # Do not throw away the model in case the user stops the training process
            pass

        self.loaded_trained = True
        # list all data in history
        if (plots):
            print(self.history.history.keys())
            self.plot_history(dir_plots)
        return self.model


