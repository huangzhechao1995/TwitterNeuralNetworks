# TwitterNeuralNetworks  

This repository is used for training neural networks for twitter sentiment analysis.  

## Dependencies  
Python 3  
Keras  
Gensim  


## Training  

Using `train.py` to train the model.   

### The usage of the variable `preprocessed` 
The training is divided into data preprocessing and training of the neural nets.   
The data preprocessing constitutes most of the runnning time of this `train.py` file. So, I added a variable `preprocessed` to the file. 
- Set `preprocessed=True` when training on new data. It will process the csv file into npy files, and then train the model. 
- Set `preprocessed=False` when training on preprocessed data. It will load the npy files, and train the model starting there.  
### Weights of the model
Weights of the model will be saved in a folder under the repository's root named `weights`

### Data Format
In the case when `preprocess=True`, the input file should be csv format, with at least three columns `twt` and `rep\dem`, which are the twitter raw text and the label of that twitter. An example is uploaded to this repository. An example could be found in the sample datasets.

## Testing
Using `get_polarity-csv.py` to test the model. The program will load the weights trained from training, and generate prediction based on the testing set. The input for testing is similar to the input for training. 

### Interpretation of the testing result
There will be a new column appended to the csv file named  `prob_0` which is a continuous number between 0 and 1. It means the probability of the tweet in the same line to be classified as 0. If the `prob_0` value is high, that means our model strongly believe this tweet should be labelled 0 (number 0 and 1 are stands for positive and negative, which are subject to the user's preference. In the case of Immigration topic, if in the training set we use 1 to represent pro-Immigration, and 0 to represent anti-Immigration, then a high `prob_0` score, such as 0.98 suggests the model strongy believes the tweet is anti-Immigration. The testing program agrees with whatever defined in the training set). 
