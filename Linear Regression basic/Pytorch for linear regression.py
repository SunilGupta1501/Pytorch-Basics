#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch


# In[5]:


w = torch.randn(2,3, requires_grad=True)
b = torch.randn(2,requires_grad=True)
print(w.size())
print(w)
print(b.size())
print(b)


# Multiplying addition to tensor

# In[6]:


b@w+torch.randn(3,requires_grad=True)


# In[7]:


## Transpose multiplication

k = torch.randn(2, 3, requires_grad=True)
print(k)
print(k@w.t())


# Gradient in tensor
# 

# In[8]:


w.grad
print(w.grad)


# reshape

# In[9]:


w.reshape(3,-1)## -1 is use of unknow dimention of the vector


# #Linear regression implimentation (mx+b )

# In[13]:


### Lets consider inputs as a independent variables with 3 features and 5 data points

inputs = np.array([[74., 43, 65], 
                   [53, 56, 87], 
                   [87, 114, 58], 
                   [102, 43, 37], 
                   [69, 97, 89]], dtype='float32')


# targets is my dependent feature 
targets = np.array([[23], 
                    [65], 
                    [54], 
                    [13], 
                    [53]], dtype='float32')


inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

w = torch.randn(1,3, requires_grad=True)
b = torch.randn(1,requires_grad=True)

### mx+b (m slope and b is bias)

def model(x):
    return x @ w.t() + b


preds = model(inputs)


# mean square error
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Compute the loss value
loss = mse(preds, targets)

print(loss)

# Compute gradients for the loss
loss.backward()

print(w)
print(w.grad)

## update the weight and biased using the gradiend 

with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()


preds = model(inputs)

loss = mse(preds, targets)
print(loss)


# Now Let's loop the complete process for 100 epoches

# In[19]:


w = torch.randn(1,3, requires_grad=True)
b = torch.randn(1,requires_grad=True)

epoches = 10000
for i in range(epoches):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

    if i%1000==0:
        print(f'The loss value for {i}th iteration is {loss}.')


# lets see the current weight and bias value and also see the prediction value from our model.

# In[25]:


print(w)
print(b)
print("Actuall Value:")
print(targets)
print("predicted Value:")
print(model(inputs))
print(mse(model(inputs), targets))


# ## Linear regression using PyTorch built-ins

# In[31]:


import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


# In[27]:



# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# In[30]:


inputs.size()
targets.size()


# ## Dataset and DataLoader
# 
# We'll create a `TensorDataset`, which allows access to rows from `inputs` and `targets` as tuples, 
# and provides standard APIs for working with many different types of datasets in PyTorch.

# In[33]:


train_ds = TensorDataset(inputs, targets)
train_ds[0:3]


# In[34]:


# Define data loader 
## shuffle = true meaning suffering the data before taking a batch size
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# In[42]:


for xb, yb in train_dl:
    print(xb)
    print(yb)
    break


# ## nn.Linear
# 
# Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.

# In[46]:


# Define model
model = nn.Linear(3,2)## (number of input values , number of output values)
print(model.weight)
print(model.bias)


# PyTorch models also have a helpful `.parameters` method, which returns a list containing all the weights and bias matrices present in the model. For our linear regression model, we have one weight matrix and one bias matrix.

# In[49]:


# Parameters
list(model.parameters())


# In[50]:


# Generate predictions
preds = model(inputs)
preds


# ## Loss Function
# 
# Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.

# In[51]:


# Import nn.functional
import torch.nn.functional as F
##
## The `nn.functional` package contains many useful loss functions and several other utilities. 
##


# In[54]:


# Define loss function
loss_fn = F.mse_loss
# Let's compute the loss for the current predictions of our model.
loss = loss_fn(model(inputs), targets)
print(loss)


# ## Optimizer
# 
# Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD is short for "stochastic gradient descent". The term _stochastic_ indicates that samples are selected in random batches instead of as a single group.

# In[55]:


# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Note that `model.parameters()` is passed as an argument to `optim.SGD` so that the optimizer knows which matrices should be modified during the update step. Also, we can specify a learning rate that controls the amount by which the parameters are modified.

# ## Train the model
# 
# We are now ready to train the model. We'll follow the same process to implement gradient descent:
# 
# 1. Generate predictions
# 
# 2. Calculate the loss
# 
# 3. Compute gradients w.r.t the weights and biases
# 
# 4. Adjust the weights by subtracting a small quantity proportional to the gradient
# 
# 5. Reset the gradients to zero
# 
# The only change is that we'll work batches of data instead of processing the entire training data in every iteration. Let's define a utility function `fit` that trains the model for a given number of epochs.

# In[61]:


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 100 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# In[62]:


fit(1000, model, loss_fn, opt, train_dl)


# In[63]:


# Generate predictions
preds = model(inputs)
preds


# In[66]:


targets


# Indeed, the predictions are quite close to our targets. We have a trained a reasonably good model.
# but we can't say this until we can see an unknow data and calculate the loss.

# In[69]:


import pandas as pd
pd.read_csv(r"C:\Users\sg185314\Downloads\train.csv")


# In[ ]:




