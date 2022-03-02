# Pruning in Pytorch

This repo consists the solution of the Bittensor Technical Challenge on Pruining a model which was trained on MNIST dataset.

### Dependencies
```
torch==1.8.0
torchvision==0.9.0
numpy
pandas
matplotlib
```
### Folder Structure
The repo consists of 3 files.
- training.ipynb
- pruning.ipynb
- mnist.pth

### Instructions

#### Training
- The code to train the classifier on the mnist dataset is present in training.ipynb
- The trained model is stored in mnist.pth
- To retrain the model and get the model file mnist.pth again you can run all the cells in the training.ipynb notebook

#### Pruning
- Before runing the pruining.ipynb notebook ensure that mnist.pth model file exists. If not train the model again using training.ipynb notebook.





