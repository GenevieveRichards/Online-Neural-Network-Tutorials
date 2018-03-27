
# coding: utf-8

# In[50]:


from __future__ import print_function
import torch
from torch.autograd import Variable
import numpy as np

# Constructing a 5 X 3 matrix (Uninitialised)
x = torch.Tensor(5,3)
#print(x)

#Constructing a randomly initialised matrix 
x = torch.rand(5,3)
#print(x)

#Size 
#print(x.size())

#Addition Syntaxes
y = torch.rand(5,3)
torch.add(x,y, out= z)
y.add_(x)
#print(y)
#print(x + y)
#print(z)

#Reshaping a Tensor 

#print(x)
#print(x[:, 1])

x = torch.rand(4,4)
y = x.view(16)
z = x.view(-1,8)  #the size -1 is inferred from other dimensions
#print(y)
#print(x)
#print(z)
# 100+ tensor operations to be found at this site http://pytorch.org/docs/master/torch.html

#Numpy to Torch Tensor 

a = torch.ones(5)
print (a)

b = a.numpy()
print(b)

a.add_(1)
print(a)
print(b)


# In[51]:


a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)


# In[52]:


#Autograd: Automatic Differentiation
x = Variable(torch.ones(2,2), requires_grad=True)
y = x + 2
z = y * y * 3
out = z.mean()
print(x)
print(y)
print(y.grad_fn)
print(z, out)


# In[53]:


#Gradients 
out.backward()
print(x.grad)


# In[54]:


x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)


# In[55]:


gradients = torch.FloatTensor([0.1, 1.0, 0.001])
y.backward(gradients)

print(x.grad)


# In[ ]:


#Further Documentation is at http://pytorch.org/docs/autograd


# In[56]:


#Define the Network - Will restart from scratch here!!
#This cell will cover the Defining the neural network points and Processing inputs and calling backward

#Packages Needed 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#defining process 


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    
net = Net()
print(net)

#The number of learnable parameters 
params = list(net.parameters())
print(len(params))
print(params[0].size())

#Input one Variable 
input = Variable(torch.randn(1,1,32,32))
out = net(input)
print (out) 

#Zero the gradient buffers of all parameters and backprops with random gradients 
net.zero_grad()
out.backward(torch.randn(1,10))


# In[60]:


#The Loss function 
#there are lots of different loss functions under the nn package.
#Using MSE - Mean Squared Error 

output = net(input)
target = Variable(torch.arange(1,11))
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

#loss.backward() #Backpropgating the Error 

#Following a few steps backward
print(loss.grad_fn) #MSELoss
print(loss.grad_fn.next_functions[0][0]) #LINEAR
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) #ReLU


# In[61]:


#BackProp

net.zero_grad()

print('Conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('Conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# In[ ]:


#The NN package contains various modules and loss functions that form the building blockas 
#of deep neural networks. http://pytorch.org/docs/master/nn.html


# In[62]:


#Updating the Weights 
# weight = weight - learning_rate * gradient 

#One Way 
learning_rate = 0.01 
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

    
#However as you update the NN you might want to use various different update rules 
#Such as SGD, Nesterov_SGD, Adam, RMSProp. To do this use torch.optim package. 


# In[63]:


import torch.optim as optim 

#Create your optimizer 
optimizer = optim.SGD(net.parameters(), lr=0.01)

#In your training Loop 
optimizer.zero_grad() #Zero the gradiet buffers 
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() #Does Update


# In[67]:


#TRAINING A CLASSIFIER

#Images - Pillow, OpenCV 
#Audio - scipy, librosa 
#text - raw Python or Cython, NLTK, SpaCy 

#Vision - created a package called torchvision - data loaders for common datasets 
#such as ImageNet, CIFAR10, MNIST and data transformers for images, viz 
#torchvision.datasets and torch.utils.data.DataLoader 


import torch
import torchvision
import torchvision.transforms as transforms

#Load and Normalised CIFAR data 
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data',train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)


testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# In[69]:


## Showing training images 

import matplotlib.pyplot as plt
import numpy as np

#functions to show an image 

def imshow(img):
    img = img / 2 + 0.5 #unnormalise
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))

#Get some random traning images 
dataiter = iter(trainloader)
images, labels = dataiter.next()

#Show Images 
imshow(torchvision.utils.make_grid(images))

#print labels 
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[76]:


#Define A Convolution Neural Network 

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)


# In[77]:


#Define a Loss Function and Optimizer 

import torch.optim as optim 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# In[79]:


#Train the Network!! 

for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        #get inputs 
        inputs, labels = data
        #Wrap them in a variable 
        inputs, labels = Variable(inputs), Variable(labels)
        
        #Zero the parameter gradiets 
        optimizer.zero_grad()
        
        #forward + backward + optimize 
        outputs = net(inputs)
        loss =criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #print Statistics 
        running_loss += loss.data[0]
        if i % 2000 == 1999: #print every 2000 mini-batches
            print('[%d, %5d] loss: % 3f' %(epoch+1, i + 1, running_loss/2000))
            running_loss = 0.0
print('Finished Training')


# In[80]:


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))


# In[82]:


outputs = net(Variable(images))

#Higher the energy for a class, the more the network thinks that the image 
#is of the particular class. Lets get the index of the highest energy 
_, predicted = torch.max(outputs.data, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# In[83]:


correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


# In[84]:


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    c = (predicted == labels).squeeze()
    for i in range(4):
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

