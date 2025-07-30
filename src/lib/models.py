import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
from . import conf

class Simple_3_Layer_CNN(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, input_size=conf.required_image_size):
        super(Simple_3_Layer_CNN, self).__init__()
        self._kernel_size1 = 3
        self._stride1 = 1
        self._padding1 = 1
        self._kernel_size2 = 3
        self._stride2 = 1
        self._padding2 = 1

        self._conv1 = nn.Conv2d(input_channels, 4, kernel_size=self._kernel_size1, stride=self._stride1, padding=self._padding1)
        self._pool1 = nn.MaxPool2d(2, 2)
        self._conv2 = nn.Conv2d(4, 8, kernel_size=self._kernel_size2, stride=self._stride2, padding=self._padding2)
        self._pool2 = nn.MaxPool2d(2,2)
        self._conv3 = nn.Conv2d(8, 16, kernel_size=self._kernel_size2, stride=self._stride2, padding=self._padding2)
        self._pool3 = nn.MaxPool2d(2,2)
        
        out_height1 = self._calculate_output_dimension(input_size[0], self._kernel_size1, self._stride1, self._padding1)
        out_width1 = self._calculate_output_dimension(input_size[1], self._kernel_size1, self._stride1, self._padding1)
        out_height2 = self._calculate_output_dimension(out_height1 // 2, self._kernel_size2, self._stride2, self._padding2)
        out_width2 = self._calculate_output_dimension(out_width1 // 2, self._kernel_size2, self._stride2, self._padding2)
        self._fc_size_1 = out_height2 * out_width2 #fully connected layer
        self._fc1 = nn.Linear(self._fc_size_1, output_channels)
        
    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1
    
    def forward(self, x, camera_length, photon_energy):
        x = self._conv1(x)
        x = F.relu(x)
        x = self._pool1(x)
        
        x = self._conv2(x)
        x = F.relu(x)
        x = self._pool2(x)
        
        x = self._conv3(x)
        x = F.relu(x)
        x = self._pool3(x)
        
        x = x.view(x.size(0), -1)
        x = self._fc1(x)
        x = F.relu(x)
        return x
    
#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#
class CNN_with_Optunas_Best(nn.Module): #
    def __init__(self, input_channels=1, output_channels=1, input_size=conf.required_image_size):
        super(CNN_with_Optunas_Best, self).__init__()
        self._output_channels = output_channels
        
        self._conv1 = nn.Conv2d(input_channels, conf.conv_channel_size, kernel_size=conf.conv_kernel_size, stride=1, padding=1)
        self._bn2d_1 = nn.BatchNorm2d(conf.conv_channel_size, momentum = conf.batch_norm_2d_momentum)
        self._conv2 = nn.Conv2d(conf.conv_channel_size, 2*conf.conv_channel_size, kernel_size=conf.conv_kernel_size, stride=1, padding=1)
        self._bn2d_2 = nn.BatchNorm2d(2*conf.conv_channel_size, momentum = conf.batch_norm_2d_momentum)
        self._conv3 = nn.Conv2d(2*conf.conv_channel_size, 4*conf.conv_channel_size, kernel_size=conf.conv_kernel_size, stride=1, padding=1)        
        self._bn2d_3 = nn.BatchNorm2d(4*conf.conv_channel_size, momentum = conf.batch_norm_2d_momentum)
        
        self._pool = nn.MaxPool2d(2, 2)
         
        #after first conv and pool        
        out_height_conv1 = self._calculate_output_dimension_after_conv(input_size[0], conf.conv_kernel_size, 1, 1)
        out_width_conv1 = self._calculate_output_dimension_after_conv(input_size[1], conf.conv_kernel_size, 1, 1)
        
        out_height_pool1 = self._calculate_output_dimension_after_pool(out_height_conv1, 2, 2)
        out_width_pool1 = self._calculate_output_dimension_after_pool(out_width_conv1, 2, 2)

        #after second conv and pool
        out_height_conv2 = self._calculate_output_dimension_after_conv(out_height_pool1, conf.conv_kernel_size, 1, 1)
        out_width_conv2 = self._calculate_output_dimension_after_conv(out_width_pool1, conf.conv_kernel_size, 1, 1)
        out_height_pool2 = self._calculate_output_dimension_after_pool(out_height_conv2, 2, 2)
        out_width_pool2 = self._calculate_output_dimension_after_pool(out_width_conv2, 2, 2)
        
        #after thrid conv and pool
        out_height_conv3 = self._calculate_output_dimension_after_conv(out_height_pool2, conf.conv_kernel_size, 1, 1)
        out_width_conv3 = self._calculate_output_dimension_after_conv(out_width_pool2, conf.conv_kernel_size, 1, 1)
        out_height_pool3 = self._calculate_output_dimension_after_pool(out_height_conv3, 2, 2)
        out_width_pool3 = self._calculate_output_dimension_after_pool(out_width_conv3, 2, 2)
        
        self._first_fc_size_input = out_height_pool3 * out_width_pool3 * conf.conv_channel_size *4 #fully connected layer # times 4 because of the output of self._conv3
        self._last_fc_size_input = conf.linear_layer_size
        self._mid_fc_size_input = 4 * conf.linear_layer_size
        
        self._dropout = nn.Dropout(conf.dropout_probability)
        
        
        if conf.num_linear_dropout_layers == 1:
            print("Setting variables with 1 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._output_channels) 
            
        if conf.num_linear_dropout_layers == 2:
            print("Setting variables with 2 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._last_fc_size_input) 
            self._bn1d_1 = nn.BatchNorm1d(self._last_fc_size_input, momentum = conf.batch_norm_1d_momentum)
            self._fc2 = nn.Linear(self._last_fc_size_input, self._output_channels)
            
        if conf.num_linear_dropout_layers == 3:
            print("Setting variables with 3 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._mid_fc_size_input) 
            self._bn1d_1 = nn.BatchNorm1d(self._mid_fc_size_input, momentum = conf.batch_norm_1d_momentum)
            self._fc2 = nn.Linear(self._mid_fc_size_input, self._last_fc_size_input)
            self._bn1d_2 = nn.BatchNorm1d(self._last_fc_size_input, momentum = conf.batch_norm_1d_momentum)
            self._fc3 = nn.Linear(self._last_fc_size_input, self._output_channels)
            

    def _calculate_output_dimension_after_conv(self, input_dim, kernel_size, stride, padding):
        return ((input_dim - kernel_size + padding*2) // stride) + 1 

    def _calculate_output_dimension_after_pool(self, input_dim, kernel_size, stride):
        return ((input_dim - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        x = self._conv1(x)
        x = F.relu(x)
        x = self._bn2d_1(x)
        x = self._pool(x)
        
        x = self._conv2(x)
        x = F.relu(x)
        x = self._bn2d_2(x)
        x = self._pool(x)
        
        x = self._conv3(x)
        x = F.relu(x)
        x = self._bn2d_3(x)
        x = self._pool(x)
        
        x = x.view(x.size(0), -1) #reshaping it to a vector
        
        
        #always at least one fc
        x = self._fc1(x) 
        x = F.relu(x)
        x = self._dropout(x)

        if (conf.num_linear_dropout_layers - 1) != 0: # if the number of layers is 2 or 3, it will not be zero
            x = self._bn1d_1(x)
            x = self._fc2(x) 
            x = F.relu(x)
            x = self._dropout(x)
            
        if conf.num_linear_dropout_layers == 3: # if the number of layers is 3, it will not be zero
            x = self._bn1d_2(x)
            x = self._fc3(x)
            x = F.relu(x)
            x = self._dropout(x)
        
        x = F.relu(x) # tbh i dont think this is necessary but im not sure and i feel like it should be so here we are    
        return x

#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#*#


#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#
class Optuna_Simple_CNN(nn.Module): #
    def __init__(self, input_channels=1, output_channels=1, input_size=conf.required_image_size, hpd=None):
        super(Optuna_Simple_CNN, self).__init__()
        self._output_channels = output_channels
        self._hpd = hpd
        if hpd is None:
            raise ValueError("Hyperparameter dictionary (hpd) cannot be None -av")        
        # needed in models.py
        self._conv_channel_size = self._hpd['conv_channel_size'] 
        self._conv_kernel_size = self._hpd['conv_kernel_size']
        self._num_linear_dropout_layers = self._hpd['num_linear_dropout_layers']
        self._linear_layer_size = self._hpd['linear_layer_size']
        self._dropout_probability = self._hpd['dropout_probability'] 
        self._momentum_2d = self._hpd['momentum_2d']
        self._momentum_1d = self._hpd['momentum_1d']
        
        self._stride = 1
        self._padding = 1
        self._pool_kernel_size = 2

        self._conv1 = nn.Conv2d(input_channels, self._conv_channel_size, kernel_size=self._conv_kernel_size, stride=self._stride, padding=self._padding)
        self._bn2d_1 = nn.BatchNorm2d(self._conv_channel_size, momentum = self._momentum_2d)
        self._conv2 = nn.Conv2d(self._conv_channel_size, 2*(self._conv_channel_size), kernel_size=self._conv_kernel_size, stride=self._stride, padding=self._padding)
        self._bn2d_2 = nn.BatchNorm2d((2*self._conv_channel_size), momentum = self._momentum_2d)
        self._conv3 = nn.Conv2d(2*(self._conv_channel_size), 4*(self._conv_channel_size), kernel_size=self._conv_kernel_size, stride=self._stride, padding=self._padding)        
        self._bn2d_3 = nn.BatchNorm2d((4*self._conv_channel_size), momentum = self._momentum_2d)
        
        self._pool = nn.MaxPool2d(self._pool_kernel_size, self._pool_kernel_size)
         
        #after first conv and pool        
        out_height_conv1 = self._calculate_output_dimension_after_conv(input_size[0], self._conv_kernel_size, self._stride, self._padding)
        out_width_conv1 = self._calculate_output_dimension_after_conv(input_size[1], self._conv_kernel_size, self._stride, self._padding)
        
        out_height_pool1 = self._calculate_output_dimension_after_pool(out_height_conv1, self._pool_kernel_size, self._stride*2)
        out_width_pool1 = self._calculate_output_dimension_after_pool(out_width_conv1, self._pool_kernel_size, self._stride*2)

        #after second conv and pool
        out_height_conv2 = self._calculate_output_dimension_after_conv(out_height_pool1, self._conv_kernel_size, self._stride, self._padding)
        out_width_conv2 = self._calculate_output_dimension_after_conv(out_width_pool1, self._conv_kernel_size, self._stride, self._padding)
        out_height_pool2 = self._calculate_output_dimension_after_pool(out_height_conv2, self._pool_kernel_size, self._stride*2)
        out_width_pool2 = self._calculate_output_dimension_after_pool(out_width_conv2, self._pool_kernel_size, self._stride*2)
        
        #after thrid conv and pool
        out_height_conv3 = self._calculate_output_dimension_after_conv(out_height_pool2, self._conv_kernel_size, self._stride, self._padding)
        out_width_conv3 = self._calculate_output_dimension_after_conv(out_width_pool2, self._conv_kernel_size, self._stride, self._padding)
        out_height_pool3 = self._calculate_output_dimension_after_pool(out_height_conv3, self._pool_kernel_size, self._stride*2)
        out_width_pool3 = self._calculate_output_dimension_after_pool(out_width_conv3, self._pool_kernel_size, self._stride*2)
        
        self._first_fc_size_input = out_height_pool3 * out_width_pool3 * self._conv_channel_size *4 #fully connected layer # times 4 because of the output of self._conv3
        self._last_fc_size_input = self._linear_layer_size
        self._mid_fc_size_input = 4 * self._linear_layer_size
        
        self._dropout = nn.Dropout(self._dropout_probability)
        
        
        if self._num_linear_dropout_layers == 1:
            print("Setting variables with 1 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._output_channels) 
            
        if self._num_linear_dropout_layers == 2:
            print("Setting variables with 2 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._last_fc_size_input) 
            self._bn1d_1 = nn.BatchNorm1d(self._last_fc_size_input, momentum = self._momentum_1d)
            self._fc2 = nn.Linear(self._last_fc_size_input, self._output_channels)
            
        if self._num_linear_dropout_layers == 3:
            print("Setting variables with 3 linear & dropout layer")
            self._fc1 = nn.Linear(self._first_fc_size_input, self._mid_fc_size_input) 
            self._bn1d_1 = nn.BatchNorm1d(self._mid_fc_size_input, momentum = self._momentum_1d)
            self._fc2 = nn.Linear(self._mid_fc_size_input, self._last_fc_size_input)
            self._bn1d_2 = nn.BatchNorm1d(self._last_fc_size_input, momentum = self._momentum_1d)
            self._fc3 = nn.Linear(self._last_fc_size_input, self._output_channels)
            

    def _calculate_output_dimension_after_conv(self, input_dim, kernel_size, stride, padding):
        return ((input_dim - kernel_size + padding*2) // stride) + 1 

    def _calculate_output_dimension_after_pool(self, input_dim, kernel_size, stride):
        return ((input_dim - kernel_size) // stride) + 1

    def forward(self, x, camera_length, photon_energy):
        x = self._conv1(x)
        x = F.relu(x)
        x = self._bn2d_1(x)
        x = self._pool(x)
        
        x = self._conv2(x)
        x = F.relu(x)
        x = self._bn2d_2(x)
        x = self._pool(x)
        
        x = self._conv3(x)
        x = F.relu(x)
        x = self._bn2d_3(x)
        x = self._pool(x)
        
        x = x.view(x.size(0), -1) #reshaping it to a vector
        
        
        #always at least one fc
        x = self._fc1(x) 
        x = F.relu(x)
        x = self._dropout(x)

        if (self._num_linear_dropout_layers - 1) != 0: # if the number of layers is 2 or 3, it will not be zero
            x = self._bn1d_1(x)
            x = self._fc2(x) 
            x = F.relu(x)
            x = self._dropout(x)
            
        if self._num_linear_dropout_layers == 3: # if the number of layers is 3, it will not be zero
            x = self._bn1d_2(x)
            x = self._fc3(x)
            x = F.relu(x)
            x = self._dropout(x)
        
        x = F.relu(x) # tbh i dont think this is necessary but im not sure and i feel like it should be so here we are    
        return x