import numpy as np
import cv2
import pandas as pd



def load_data(path_csv, path_img):
    '''
    This function takes as an argument the paths for the csv training/testing file as well as the path for the images, and 
    returns x and y
    
    Input : -'path_csv' : path that links to the exact csv file (must finish by '.csv')
            -'path_img' : path that links to the repository where all the images are stored
            -'size_img' : desired size (not counting the RGB channels) for the imported images (original size is 640x480)
    
    Output : -'x' : array containing the image representation for each image 
             -'y' : array of labels associated to x_train arrays
    '''
    #initialize x_train and y_train
    x=[]
    y=[]
    
    #load the dataframe
    train_data = pd.read_csv(path_csv)
    
    #loop over all rows of this dataframe to construct x_train and y_train
    for ind, row in train_data.iterrows():
        img = cv2.imread(path_img+row['name']) #load the image
        img = img/255.0 #normalizing the pixel values
        img = img.astype('float32') #converting the type of pixel to float 32
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to gray scale
        
        #add the array representing the image into x_train and the label into y_train
        x.append(gray)
        y.append(row['label'])
        
    #convert to arrays
    x = np.array(x)
    y = np.array(y)
        
    return x, y