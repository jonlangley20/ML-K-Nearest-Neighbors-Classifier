#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# ## Question 1

# In[2]:


import pandas as pd
import numpy as np
from math import sqrt
from scipy.stats import mode


# In[5]:


train = open("pa1train.txt","r")
X_train = []
y_train = []
for line in train.readlines():
    vec = line.split(' ')
    X_train.append(vec[:-1])
    y_train.append(vec[-1][0])
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_train
y_train


# In[174]:


val = open("pa1validate.txt","r")
X_val = []
y_val = []
for line in val.readlines():
    vec = line.split(' ')
    X_val.append(vec[:-1])
    y_val.append(vec[-1][0])
X_val = pd.DataFrame(X_val)
y_val = pd.DataFrame(y_val)


# In[175]:


test = open('pa1test.txt', 'r')
X_test = []
y_test = []
for line in test.readlines():
    vec = line.split(' ')
    X_test.append(vec[:-1])
    y_test.append(vec[-1][0])
    
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)


# In[176]:


def euclidean_distance(row1, row2):
    distance = 0.0
    
    distance = np.linalg.norm(row1 - row2)
    
    
    return distance
   
    


# In[177]:


print(X_train)
y_train


# In[178]:


#def convert_to_int(X_train, X_val, X_test):
    #all values in the txt file are strings, must convert them to numerics, converting to
    #int instead of float saves numerous space
for i in X_train.columns:
    X_train[i] = X_train[i].astype(int)
for i in X_val.columns:
    X_val[i] = X_val[i].astype(int)
for i in X_test.columns:
    X_test[i] = X_test[i].astype(int)
   # return X_train, X_val, X_test
    
#convert_to_int(X_train, X_val, X_test)


# In[179]:



#def convert_tables_to_np(X_train,X_val,X_test, y_train,y_val,y_test):
    #convert integer columns into numpy arrays, matrix manipulation drastically decreases
    #run time, equalling far faster run time
X_train = X_train.to_numpy()
X_val = X_val.to_numpy()
X_test = X_test.to_numpy()

y_train = y_train.iloc[:, 0]
y_train= y_train.to_numpy()
y_val = y_val.iloc[:, 0]
y_val= y_val.to_numpy()
y_test = y_test.iloc[:, 0]
y_test= y_test.to_numpy()
  #  return X_train, X_val, X_test, y_train, y_val,y_test
#convert_tables_to_np(X_train,X_val,X_test, y_train,y_val,y_test)


# In[ ]:




        


# In[183]:


def get_neighbors(training_data, potential_labels, input_data, k_amount):
    arr_of_predicts = []
    input_size = len(input_data)
    training_size = len(training_data)
    for i in range(input_size):
        arr_of_dist = []
        for j in range(training_size):
            arr_of_dist.append((potential_labels[j], euclidean_distance(input_data[i], training_data[j])))
        arr_of_dist.sort(key = lambda item: item[1])
        k_closest_values = arr_of_dist[:k_amount]
        actual_labels = []
        for l in k_closest_values:
            actual_labels.append(l[0])
        arr_of_predicts.append(mode(actual_labels)[0][0])
    arr_of_predicts = np.array(arr_of_predicts)
    return arr_of_predicts
        


# In[181]:


def NNs_pred(training_set, labels, x, k):
    preds = []
    for i in range(len(x)):
        distances = []
        for j in range(len(training_set)):
            distances.append((labels[j], eucdist(x[i], training_set[j])))
        distances.sort(key=lambda item: item[1])
        distances = distances[:k]
        labs = [x[0] for x in distances]
        preds.append(mode(labs)[0][0])
    return np.array(preds)


# In[185]:


train_data_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    train_data_errors[i] = np.mean(get_neighbors(X_train, y_train, X_train, i) != y_train)
    
train_data_errors


# In[188]:


val_data_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    val_data_errors[i] = np.mean(get_neighbors(X_train, y_train, X_val, i) != y_val)
    
val_data_errors


# In[209]:


#Make data table of training data and validation errors
error_df = pd.DataFrame(list(val_data_errors.items()),columns = ['K Classifier','Validate Errors'])

error_df['Train errors'] = error_df['K Classifier'].map(train_data_errors) 
error_df


# In[211]:


test_data_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    test_data_errors[i] = np.mean(get_neighbors(X_train, y_train, X_test, i) != y_test)
    
test_data_errors


# The classifier with the lowest error on the validation data is the k parameter value of 1, which yields a test error of about 9.4%

# ## Question 2
# 

# In[192]:


projections = open('projection.txt', 'r')
project_vector_arr = []
#The data is just a text, must use this code to make it usable in numpy
#used integers before because the float wasn't necessary and took 
#additional space, but float is necessary for analysis of vector multiplication
for line in projections.readlines():
    proj_vector = line.split(' ')
    proj_vector[-1] = proj_vector[-1][:-1]
    
    proj_vector = np.array(proj_vector).astype(float)
    project_vector_arr.append(proj_vector)


# In[193]:


project_train = np.matmul(X_train, project_vector_arr)
#projection of vectors onto the column space can simply be
#accomplished through vector multiplication, and efficiently 
#accomplished through the use of numpy arrays

project_validate = np.matmul(X_val, project_vector_arr)

project_test = np.matmul(X_test, project_vector_arr)


# In[197]:


train_proj_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    train_proj_errors[i] = np.mean(get_neighbors(project_train, y_train, project_train, i) != y_train)
    
    
train_proj_errors


# In[198]:


val_proj_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    val_proj_errors[i] = np.mean(get_neighbors(project_train, y_train, project_validate, i) != y_val)
    
val_proj_errors


# In[210]:


test_proj_errors = {}
#dict will be easiest to show key: k amount and value error rate
potential_k_values = [1,3,5,9,15]
for i in potential_k_values:
    test_proj_errors[i] = np.mean(get_neighbors(project_train, y_train, project_test, i) != y_test)
    
test_proj_errors


# By projecting the input data onto the column space of the matrix, the model is affected in several ways.  A notable feature is that the accuracy of the model is significantly lowered, and I'm honestly uncertain as to the source of this.  A significant improvement is that the runtime of the model has drastically improved.  I noticed that running the test errors before projection had a runtime of about 3 minutes, while post-projection had a runtime of about 1 minute and 30 seconds, a nearly 50% decrease.

# In[ ]:




