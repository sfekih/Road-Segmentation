from cnn import Convolutional_neural_net 
from unet import Unet 
from utils.preprocess_data import preprocess, load_images_test
from make_submission import *
import os
from optparse import OptionParser


#get the arguments passed by the user
parser = OptionParser()

#using cpu or gpu
parser.add_option("-u", "--cpu", 
                  help="add --cpu if you want to use cpu othw gpu is used",
                  action="store_true", dest="cpu",
                  default=False
                  )

#get the model type
parser.add_option("-c", "--cnn", 
                  help="add --cnn if you want to run a cnn model and nothing othw",
                  action="store_true", dest="cnn",
                  default=False
                  )

#choose if you want to use paddings or not in unet
parser.add_option("-p", "--unet_608", 
                  help="If you chose unet, add --unet_608 if you want to run your model with input images of 608*608 pixels othw it works with inputs of 400*400 pixels",
                  action="store_true", dest="unet_608",
                  default=False
                  )

#choose if you want to Relu or LeakyRelu as activation function for the model
parser.add_option("-l", "--activation_Relu", 
                  help="add --activation_Relu if you want to use Relu as activation function othw it will use LeakyRelu",
                  action="store_true", dest="activation_Relu",
                  default=False
                  )
                  
#choose the batch size
parser.add_option("-s", "--batch_size", 
                  help="set the batch size, the default is 16",
                  action="store", dest="batch_size",
                  type="int",
                  default=16
                  )

#get the train data path
parser.add_option("-d", "--traindatapath", 
                  help="specify where we can find the data",
                  action="store", dest="traindatapath",
                  default=os.path.join("..", "data", "training"))
#get the test data path
parser.add_option("-t", "--testdatapath", 
                  help="specify where we can find the data",
                  action="store", dest="testdatapath",
                  default=os.path.join("..", "data", "test_set_images"))

#get the submission file path
parser.add_option("-b", "--submissiondatapath", 
                  help="specify where you want to store the submission file",
                  action="store", dest="submissiondatapath",
                  default=os.path.join("..", "submissions", "unknown"))

#get the model name
parser.add_option("-m", "--modelname", 
                  help="give a name to your model for saving",
                  action="store", dest="modelname",
                  default="unknown_model_name")

#get the number of epochs
parser.add_option("-e", "--epochs", 
                  help="Choose the number of epochs you want to run your model with",
                  action="store", dest="epochs",
                  type="int",
                  default=100
                  )
                  

#get the parses
options, _ = parser.parse_args()


#the train data spath
PATH = options.traindatapath

print("let's preprocess")
#preprocess
X_train, X_test, Y_train, Y_test = preprocess(PATH, Unet= (False if options.cnn else True))
print("done preprocessing")

INPUT_SHAPE=(X_train.shape[1],X_train.shape[2],X_train.shape[3])


#set the model
model = Convolutional_neural_net(input_shape=INPUT_SHAPE, activation_Leaky= not options.activation_Relu , gpu= not options.cpu) if options.cnn else Unet(input_shape=INPUT_SHAPE, activation_Leaky= not options.activation_Relu , gpu= not options.cpu  )
print("built the model")

#fit the model
model.fit(X_train, Y_train,batch_size=options.batch_size , epochs=options.epochs,   dir_plots=os.path.join("..", "plots", options.modelname ))
print("fit the model")

#save the model
model.serialize(name= options.modelname)
print("saved the model")

#generate the submission file and store it. Plot some predictions if requested if you convert it to an .ipynb file!
get_prediction_csv (model, options.testdatapath, submission_filename=options.submissiondatapath, 
                   Unet=(False if options.cnn else True),padd_608=options.unet_608, with_avg=False,  plot_images=True,number_images_to_plot=5)





