# TwitterNeuralNetworks  

This repository is used for training neural networks for twitter sentiment analysis.  

## Dependencies  
Python 3  
Keras  
Gensim  


## Training  
Using `train.py` to train the model.   
The training is divided into data preprocessing and training of the neural nets.   
The data preprocessing constitutes most of the runnning time of this `train.py` file. So, I added a variable `preprocessed` to the file. 
- Set `preprocessed=True` when training on new data. It will process the csv file into npy files, and then train the model. 
In this case, the input file should be csv format, with at least three columns `twt` and `rep\dem`, which are the twitter raw text and the label of that twitter. An example is uploaded to this repository.    
- Set `preprocessed=False` when training on preprocessed data. It will load the npy files, and train the model starting there.  


## Testing
Using `get_polarity-csv.py` to test the model.  
