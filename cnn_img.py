import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
import cv2
import pandas as pd


##Class implementation for the classifier
#We use the class definition from the notebook 'image_class.ipynb'

class img_class(nn.Module):
    '''  Class implenting the network that will classify the images    '''
    
    def __init__(self):
        super(img_class, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3) #first convolution layer
        self.conv2 = nn.Conv2d(6, 16, 3) #second convolution layer
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 9 * 5, 120)  # 9*5 from image dimension after the 2 convolutions+pooling (divide by 4 and substract 2 for both dimensions)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def learn(self, x_train, y_train, loss_function, optimizer, num_epochs):
        ''' Training process '''
        #convert x_train to a torch tensor
        x_train = x_train.reshape(len(x_train), 1, 44, 28) #reshape the array for torch
        tensor_x  = torch.from_numpy(x_train) #create the tensor
        tensor_x.requires_grad_()
        
        #convert y_train to a torch tensor
        tensor_y = torch.from_numpy(y_train)
        tensor_y = tensor_y.long() #convert to float
        
        #training : for each element of the input tensor 'tensor_x', compare the output of the model and the real label
        for i in range(num_epochs):
            
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.zero_grad()

            # Step 2 - compute model predictions and loss
            output = self(tensor_x) #output of the model (shape : (1050, 10))
            #pred_label = torch.argmax(output, dim=1)
            #pred_label = pred_label.float()

            #loss
            loss = loss_function(output, tensor_y) 

            # Step 3 - do a backward pass and a gradient update step
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()

            if i % 10 == 0: #print every 2 epochs
                print('Epoch : {}/{}, Loss : {}'.format(i + 1, num_epochs, loss.item()))
                
    def testing(self, x_test, y_test):
        '''Testing process'''
        #convert x_test into a tensor
        x_test = x_test.reshape(len(x_test), 1, 44, 28) #reshape the array for torch
        tensor_x  = torch.from_numpy(x_test) #create the tensor
        
        #convert y_test to a torch tensor
        tensor_y = torch.from_numpy(y_test)
        tensor_y = tensor_y.long() #convert to float
        
        output = self(tensor_x)
        softmax = torch.exp(output).cpu()
        prob = list(softmax.detach().numpy())
        predictions = np.argmax(prob, axis=1)

        # accuracy on testing set
        print(accuracy_score(y_test, predictions))
        
    def learn2(self, x_train, y_train, x_test, y_test, loss_function, optimizer, num_epochs):
        ''' Training process, second version : to have the training and testing errors '''
        train_losses = []
        test_losses = []
        
        #convert x_train to a torch tensor
        x_train = x_train.reshape(len(x_train), 1, 44, 28) #reshape the array for torch
        tensor_x  = torch.from_numpy(x_train) #create the tensor
        tensor_x.requires_grad_()
        
        #convert y_train to a torch tensor
        tensor_y = torch.from_numpy(y_train)
        tensor_y = tensor_y.long() #convert to float
        
        #convert x_test into a tensor
        x_test = x_test.reshape(len(x_test), 1, 44, 28) #reshape the array for torch
        tensor_xtest  = torch.from_numpy(x_test) #create the tensor
        #convert y_test to a torch tensor
        tensor_ytest = torch.from_numpy(y_test)
        tensor_ytest = tensor_ytest.long() #convert to float
        
        #training : for each element of the input tensor 'tensor_x', compare the output of the model and the real label
        for i in range(num_epochs):
            
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            self.zero_grad()

            # Step 2 - compute model predictions and loss
            output_train = self(tensor_x) #output of the model (shape : (1050, 10))
            output_test  = self(tensor_xtest)
            pred_label = torch.argmax(output_train, dim=1)
            pred_label = pred_label.float() #should be a number

            #loss
            loss_train = loss_function(output_train, tensor_y) 
            loss_test = loss_function(output_test, tensor_ytest)
            train_losses.append(loss_train)
            test_losses.append(loss_test)

            # Step 3 - do a backward pass and a gradient update step
            optimizer.zero_grad()  
            loss_train.backward()
            optimizer.step()

            if i % 10 == 0: #print every 2 epochs
                print('Epoch : {}/{}, Loss : {}'.format(i + 1, num_epochs, loss_train.item()))
        return train_losses, test_losses