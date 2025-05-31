import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models


#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
class Simple_3_Layer_CNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, input_size=(512, 512)):
        super(Simple_3_Layer_CNN, self).__init__()
        self.kernel_size1 = 3
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3
        self.stride2 = 1
        self.padding2 = 1

        self.conv1 = nn.Conv2d(input_channels, 4, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.pool3 = nn.MaxPool2d(2,2)
        
        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        self.fc_size_1 = out_height2 * out_width2 #fully connected layer
        self.fc1 = nn.Linear(self.fc_size_1, output_channels)
        
    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1
    
    def forward(self, x, camera_length, photon_energy):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        return x
    
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#

#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#
class Optuna_Simple_CNN(nn.Module): #
    def __init__(self, input_channels=1, output_channels=1, input_size=(512, 512), hpd=None):
        super(Optuna_Simple_CNN, self).__init__()
        self.output_channels = output_channels
        self.hpd = hpd
        if hpd is None:
            raise ValueError("Hyperparameter dictionary (hpd) cannot be None -av")        
        # needed in models.py
        self.conv_channel_size = self.hpd['conv_channel_size'] 
        self.conv_kernel_size = self.hpd['conv_kernel_size']
        self.num_linear_dropout_layers = self.hpd['num_linear_dropout_layers']
        self.linear_layer_size = self.hpd['linear_layer_size']
        self.dropout_probability = self.hpd['dropout_probability'] #not implemented yet
        
        self.stride = 1
        self.padding = 1
        self.pool_kernel_size = 2

        self.conv1 = nn.Conv2d(input_channels, self.conv_channel_size, kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.conv_channel_size, 2*(self.conv_channel_size), kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)
        self.conv3 = nn.Conv2d(2*(self.conv_channel_size), 4*(self.conv_channel_size), kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)        
        #this is fine?
        
        self.pool = nn.MaxPool2d(self.pool_kernel_size, self.pool_kernel_size)
         
        #after first conv and pool        
        out_height_conv1 = self.calculate_output_dimension_after_conv(input_size[0], self.conv_kernel_size, self.stride, self.padding)
        out_width_conv1 = self.calculate_output_dimension_after_conv(input_size[1], self.conv_kernel_size, self.stride, self.padding)
        out_height_pool1 = self.calculate_output_dimension_after_pool(out_height_conv1, self.pool_kernel_size, self.stride*2)
        out_width_pool1 = self.calculate_output_dimension_after_pool(out_width_conv1, self.pool_kernel_size, self.stride*2)

        #after second conv and pool
        out_height_conv2 = self.calculate_output_dimension_after_conv(out_height_pool1, self.conv_kernel_size, self.stride, self.padding)
        out_width_conv2 = self.calculate_output_dimension_after_conv(out_width_pool1, self.conv_kernel_size, self.stride, self.padding)
        out_height_pool2 = self.calculate_output_dimension_after_pool(out_height_conv2, self.pool_kernel_size, self.stride*2)
        out_width_pool2 = self.calculate_output_dimension_after_pool(out_width_conv2, self.pool_kernel_size, self.stride*2)
        
        #after thrid conv and pool
        out_height_conv3 = self.calculate_output_dimension_after_conv(out_height_pool2, self.conv_kernel_size, self.stride, self.padding)
        out_width_conv3 = self.calculate_output_dimension_after_conv(out_width_pool2, self.conv_kernel_size, self.stride, self.padding)
        out_height_pool3 = self.calculate_output_dimension_after_pool(out_height_conv3, self.pool_kernel_size, self.stride*2)
        out_width_pool3 = self.calculate_output_dimension_after_pool(out_width_conv3, self.pool_kernel_size, self.stride*2)
        
        self.first_fc_size_input = out_height_pool3 * out_width_pool3 * self.conv_channel_size *4 #fully connected layer # times 4 because of the output of self.conv3
        self.last_fc_size_input = self.linear_layer_size
        self.mid_fc_size_input = 4 * self.linear_layer_size
        
        self.dropout = nn.Dropout(self.dropout_probability)
        
        if self.num_linear_dropout_layers == 1:
            print("Setting variables with 1 linear & dropout layer")
            self.fc1 = nn.Linear(self.first_fc_size_input, self.output_channels) 
            
        if self.num_linear_dropout_layers == 2:
            print("Setting variables with 2 linear & dropout layer")
            self.fc1 = nn.Linear(self.first_fc_size_input, self.last_fc_size_input) 
            self.fc2 = nn.Linear(self.last_fc_size_input, self.output_channels)
            
        if self.num_linear_dropout_layers == 3:
            print("Setting variables with 3 linear & dropout layer")
            self.fc1 = nn.Linear(self.first_fc_size_input, self.mid_fc_size_input) 
            self.fc2 = nn.Linear(self.mid_fc_size_input, self.last_fc_size_input)
            self.fc3 = nn.Linear(self.last_fc_size_input, self.output_channels)
            

    def calculate_output_dimension_after_conv(self, input_dim, kernel_size, stride, padding):
        return ((input_dim - kernel_size + padding*2) // stride) + 1 

    def calculate_output_dimension_after_pool(self, input_dim, kernel_size, stride):
        return ((input_dim - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1) #reshaping it to a vector)
        
        
        #always at least one fc
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.dropout(x)

        if (self.num_linear_dropout_layers - 1) != 0: # if the number of layers is 2 or 3, it will not be zero
            x = self.fc2(x) 
            x = F.relu(x)
            x = self.dropout(x)
            
        if self.num_linear_dropout_layers == 3: # if the number of layers is 3, it will not be zero
            x = self.fc3(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        x = F.relu(x) # tbh i dont think this is necessary but im not sure and i feel like it should be so here we are    
        return x


#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#

