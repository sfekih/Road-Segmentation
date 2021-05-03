# Road segmentation

## Description 
This project is part of the ML course at EPFL. The goal is to develop a classifierthat successfully classifies roads and background segments in satellite images. We performd data preprocessing and developed two models, a Convolutional NeuralNetworks (CNN) model and a U-Net model then compared them.  An F1-score of 0.85 was achieved using the U-net model on the train set and a F1-score of 0.886 on the test set.


## Instructions 

We provide the `requirements.txt` file which allows to install dependencies as follows:

```bash
$ pip install -r requirements.txt
```
You can run the models as follows:
```bash 
git clone <repo_url>  #clone the repo
cd src
python run.py #for Unet architecture (The one used for the submission)
#python run.py -cnn  # for cnn architecutre

#use the --help to see the arguments that you can enter
  ```
  
## Dataset 
The dataset is available from the https://www.aicrowd.com/challenges/epfl-ml-road-segmentation page
It contains: 
* **training set:** 100 RGB images of dimension (400,400,3) with 100 ground truth images (black and white) of dimensions (400,400,1)
* **test set:** 50 RGB images of dimension (608,608,3)  

include the dataset in the "data" folder

## Project structure
```
├── data                            # data directory
|   ├── training
|       |__ groundtruth             # Directory containing the train groundtruth
|       |__ images                  # Directory containing the train images
|   ├── test
|       |__ images                  # Directory containing the test images
├── src                			        # Source code directory
|   ├── utils
|       |__ data_augmentation.py    # file where we perform data augmentation
|       |__ load_helpers.py         # Helper functions used in our project
|       |__ metrics.py              # file where we define the metrics (F1 score, precision, recall)
|       |__ preprocess_data.py      # helper functions for preprocessing data 
│   ├── run.py                      # script to run on terminal and run the models or train new ones
│   ├── make_submission.py          # file that contains one function called to make predictions
│   ├── model.py                    # abstract class for the models
│   ├── Unet.py                     # file where we code our Unet model
│   ├── cnn.py                      # file where we code our CNN model
├── plots              	            # Directory containing plots of metrics after running models
├── history              	          # Directory giving th evolution of the metrics of models
├── submissions          	          # Directory csv documents of main submisisons we did on Aicrowd
├── trained models                  # Directory containing some already build and trained models
├── requirements.txt			          # Contains requirements to be installed using pip
├── project2_description.pdf			  # pdf file of project description
├── requirements.txt			          # Contains requirements to be installed before running the project
├── LICENSE 
├── report.pdf			                # Contains the report of our project
└── README.md
```
**Note:** We didn't include the data folder but we kept it in the architecture to show how it should be when you add the data.  

## Trained models:
We included only 3 of our trained models as examples: 
* 'cnn_new_aug_200_16_leaky': CNN model, Leaky-Relu, 200 epochs,  
* 'unet_new_augg_400_batch4_300ep': Unet model, Leaky-Relu , 300 epochs, batch size: 4, image size: 400
* 'unet_new_augg_608_batch4_300ep.csv': Unet model, Leaky-Relu , 300 epochs, batch size: 4, image size: 608  

**Note:** The 'unet_new_augg_608_batch4_300ep.csv' is the the one that gave the best results and with whom the submission was made.    

**Note** We included the history, submission and plots for these models

## Authors 
Zeineb Ayadi  
Selim Fekih  
Nizar Ghandri   

 
