import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models



class Binary_Classification(nn.Module):
    def __init__(self, input_channels=1, output_channels=3, input_size=(2163, 2069)):
        super(Binary_Classification, self).__init__()
  
        self.kernel_size1 = 10
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3
        self.stride2 = 1
        self.padding2 = 1
        num_groups1 = 4  
        num_groups2 = 4  

        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=8)
        self.pool1 = nn.MaxPool2d(2, 2) 
        self.conv2 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=16)

        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)  # Pooling reduces size
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        
        self.fc_size = 16 * out_height2 * out_width2
        self.fc = nn.Linear(self.fc_size, output_channels)

    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1

    def forward(self, x):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(-1, self.fc_size) 
        x = self.fc(x)
        return x
    
    
class Binary_Classification_With_Parameters(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, input_size=(512, 512)):
        super(Binary_Classification_With_Parameters, self).__init__()
        self.kernel_size1 = 10
        self.stride1 = 1
        self.padding1 = 1
        self.kernel_size2 = 3
        self.stride2 = 1
        self.padding2 = 1
        num_groups1 = 4
        num_groups2 = 4
        self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=self.kernel_size1, stride=self.stride1, padding=self.padding1)
        self.gn1 = nn.GroupNorm(num_groups=num_groups1, num_channels=8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=self.kernel_size2, stride=self.stride2, padding=self.padding2)
        self.gn2 = nn.GroupNorm(num_groups=num_groups2, num_channels=16)
        out_height1 = self.calculate_output_dimension(input_size[0], self.kernel_size1, self.stride1, self.padding1)
        out_width1 = self.calculate_output_dimension(input_size[1], self.kernel_size1, self.stride1, self.padding1)
        out_height2 = self.calculate_output_dimension(out_height1 // 2, self.kernel_size2, self.stride2, self.padding2)
        out_width2 = self.calculate_output_dimension(out_width1 // 2, self.kernel_size2, self.stride2, self.padding2)
        self.fc_size_1 = 16 * out_height2 * out_width2
        self.fc_size_2 = (out_height2 * out_width2) // 23782
        self.fc1 = nn.Linear(self.fc_size_1, self.fc_size_2)
        self.fc2 = nn.Linear(self.fc_size_2 + 2, output_channels)
    def calculate_output_dimension(self, input_dim, kernel_size, stride, padding):
        return ((input_dim + 2 * padding - kernel_size) // stride) + 1
    def forward(self, x, camera_length, photon_energy):
        x = self.pool1(F.relu(self.gn1(self.conv1(x))))
        x = F.relu(self.gn2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        device = x.device
        camera_length = camera_length.to(device).float()
        photon_energy = photon_energy.to(device).float()
        params = torch.stack((camera_length, photon_energy), dim=1) #error 12/17 3:33, but it seems to be bc cam_len & phot_en are empty
        x = torch.cat((x, params), dim=1)
        x = self.fc2(x)
        return x

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

        self.conv1 = nn.Conv2d(input_channels, self.conv_channel_size, kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)
        self.conv2 = nn.Conv2d(self.conv_channel_size, 2*(self.conv_channel_size), kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)
        self.conv3 = nn.Conv2d(2*(self.conv_channel_size), 4*(self.conv_channel_size), kernel_size=self.conv_kernel_size, stride=self.stride, padding=self.padding)        
        #this is fine?
        
        self.pool = nn.MaxPool2d(2, 2)
        self.pool_kernel_size = 2 
        #after first conv and pool        
        out_height_conv1 = self.calculate_output_dimension_after_conv(input_size[0], self.conv_kernel_size, self.stride, self.padding)
        out_width_conv1 = self.calculate_output_dimension_after_conv(input_size[1], self.conv_kernel_size, self.stride, self.padding)
        out_height_pool1 = self.calculate_output_dimension_after_pool(out_height_conv1, self.pool_kernel_size, self.stride*2)
        out_width_pool1 = self.calculate_output_dimension_after_pool(out_width_conv1, self.pool_kernel_size, self.stride*2)
        #above here is correct i believe
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
        
        
        self.fc_size = out_height_pool3 * out_width_pool3 * self.conv_channel_size *4 #fully connected layer #i think that 0.25 is conv_channel_size_1/output_channels.... based on the other cnn
        self.fc = nn.Linear(self.fc_size, output_channels)

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
        
        #TODO: add dropout and linear layer
        x = x.view(x.size(0), -1) #reshaping it to a vector)
        x = self.fc(x) 
        x = F.relu(x)
        return x


#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#?#


class Binary_Classification_DenseNet(nn.Module):
    def __init__(self):
        super(Binary_Classification_DenseNet, self).__init__()
        self.densenet = torchvision.models.densenet121(pretrained=True)
        
        # Modify the first convolutional layer to accept 1-channel input instead of 3
        self.densenet.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        num_features = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Identity()  # Remove the original classifier
        
        self.fc1 = nn.Linear(num_features + 2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x, camera_length, photon_energy):
        # Extract features from the DenseNet
        features = self.densenet.features(x)
        x = F.relu(features, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(features.size(0), -1)
        
        # Concatenate additional parameters
        params = torch.stack((camera_length, photon_energy), dim=1)
        x = torch.cat((x, params), dim=1)
        
        # Pass through the modified classifier
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        
        return x
