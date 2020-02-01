import torch.nn as nn
import torch.nn.functional as F

# define the CNN architecture
class Net(nn.Module):
    ### TODO: choose an architecture, and complete the class
    #classifier=Sequential()

    def __init__(self):
        super(Net, self).__init__()   
        
        #here we used to identify a CNN with 5 layers
        self.conv1=nn.Conv2d(3,16,3)
        self.conv2=nn.Conv2d(16,32,3)
        self.conv3=nn.Conv2d(32,64,3)
        self.conv4=nn.Conv2d(64,128,3)
        self.conv5=nn.Conv2d(128,256,3)
        
        #we inizialize linearify the netowk 
        self.fc1=nn.Linear(256*6*6,133)
        
        #We inizialize max_pool the network
        self.max_pool=nn.MaxPool2d(2,2,ceil_mode=True)

        # we inizialize the dropout of the network
        self.dropout=nn.Dropout(0.25)
        
        #we inizialize 5 normalized batches 
        self.conv_bn1 = nn.BatchNorm2d(224,3)
        self.conv_bn2 = nn.BatchNorm2d(16)
        self.conv_bn3 = nn.BatchNorm2d(32)
        self.conv_bn4 = nn.BatchNorm2d(64)
        self.conv_bn5 = nn.BatchNorm2d(128)
        self.conv_bn6 = nn.BatchNorm2d(256)


    def forward(self, x):
    
        #we inizialize a backward function for each layer
        #first we rectify each layer in this network
        #then we max pool this layer which had been rectified and 
        #the result of this process we added to in a batch normalization
        # we repeate this process for 4 times after the first time
        
        x = F.relu(self.conv1(x))
        x = self.max_pool(x)
        x = self.conv_bn2(x)

        x = F.relu(self.conv2(x))
        x = self.max_pool(x)
        x = self.conv_bn3(x)

        x = F.relu(self.conv3(x))
        x = self.max_pool(x)
        x = self.conv_bn4(x)

        x = F.relu(self.conv4(x))
        x = self.max_pool(x)
        x = self.conv_bn5(x)

        x = F.relu(self.conv5(x))
        x = self.max_pool(x)
        x = self.conv_bn6(x)
        # we reshape the result of the layer
        x = x.view(-1, 256 * 6 * 6)
        
        #then we create a dropout layer
        x = self.dropout(x)
        x = self.fc1(x)

        
        #return the result of the network
        return x


# instantiate the CNN
model_scratch = Net()

# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
