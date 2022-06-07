#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# In[2]:


"""
    If GPU is available then run on GPU else run on CPU
"""
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev) 


# In[3]:


"""
    . torchvision.datasets.MNIST downloads PIL images 
    . Hence we need to define transform object to convert downloaded MNIST PIL images to Tensors
    . Batch size of 32 was used. 
    
    Reference --> https://pytorch.org/vision/stable/datasets.html#mnist
    
"""
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

# Creating Data Loader object for creating the Training set by using the field train=True  

train_data = torchvision.datasets.MNIST("./",train=True,download=True ,transform=transform)
train_data_loaded = torch.utils.data.DataLoader(train_data,
                                          batch_size=32,
                                          shuffle=True)


# Creating Data Loader object for creating the Test set by using the field train=False 
test_data = torchvision.datasets.MNIST("./",train=False,download=True,transform=transform)
test_data_loaded = torch.utils.data.DataLoader(test_data,
                                          batch_size=32,
                                          shuffle=True)


# In[4]:


"""
Creating the Model

input (inp) ==> 28*28 neurons
layer1      ==> 1000 neurons
layer2      ==> 1000 neurons
layer3      ==> 500 neurons
layer4      ==> 200 neurons
output(out) ==> 10 neurons  Since number of classes are 10

* Softmax layer is not being used as the final layer becuase Crossentropy loss
  is being used which takes the input as logits and calculates softmax before computing loss. 
  
  Reference --> https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

class Model(torch.nn.Module):
    
    # Intialising the number of neurons in the layers
    def __init__(self):
        super(Model,self).__init__()
        self.inp = nn.Linear(28*28,1000)
        self.layer1 = nn.Linear(1000,1000)
        self.layer2 = nn.Linear(1000,1000)
        self.layer3 = nn.Linear(1000,500)
        self.layer4 = nn.Linear(500,200)
        self.out = nn.Linear(200,10)
    
    # Defining the forward pass of the model
    
    def forward(self,x):
        x = torch.flatten(x,1)
        x = self.inp(x)
        x = F.relu(x)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.layer4(x)
        x = F.relu(x)
        x = self.out(x)
        #x = F.softmax(x,dim = 1)  
        return x


# In[5]:


model = Model()
model.to(torch.device("cuda:0"))                   # Loading the model to GPU is avalible else CPU


# In[6]:


"""
   1) Cross Entropy loss is being used because this is a classification model.
   2) Softmax of the ouput logits is first calculated and then classified to calculate the loss 
   3) Adam Optimizer is used. 
"""
criterion = torch.nn.CrossEntropyLoss()            # initializing Loss function
optimizer = torch.optim.Adam(model.parameters())   # initializing Adam Optimizer 


# In[7]:


"""
   Train the model for 50 epochs and printing loss for each batch and epoch.
"""
epochs = 50                                        # number of epochs


for epoch in range(epochs):                        # iterate over number of epochs
    running_loss = 0.0
    for i,images in enumerate(train_data_loaded):  # iterate over the batches of training data
        batch_images,batch_labels = images
        batch_images = batch_images.to(device)     
        batch_labels = batch_labels.to(device)
        optimizer.zero_grad()  
        outputs = model(batch_images)              # computing forward pass 
        loss = criterion(outputs, batch_labels)    # Calculating loss
        loss.backward()                            # Calculating the gradients
        optimizer.step()                           # Updating the weights
        
        """
            Code below for printing of Losses for each batch and epoch was refered from the pytorch Documentation
            https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            
        """
        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0


# In[8]:


"""
   Saving the trained model
"""
torch.save(model.state_dict(), "mnist.pth")

