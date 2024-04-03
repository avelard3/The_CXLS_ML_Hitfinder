import torch
import torch.nn as nn
import torch.optim as optim
import models as m


class Get_Configuration_Details():
    
    def __init__(self):
        pass
        
    def get_model(self) -> nn.Module:
        return self._model
    
    def get_criterion(self) -> nn.Module:
        return self._criterion
    
    def get_feature(self) -> str:
        return self._feature
    
    def get_classes(self) -> int:
        return self._classes
    
    def get_labels(self) -> list:
        return self._labels
    
    def get_formatted_image_attributes(self, image_attribute: torch.Tensor) -> torch.Tensor:
        return self._formatted_image_attribute
    
    
class Peak_Detection_Configuration(Get_Configuration_Details):
    
    def __init__(self): 
        super().__init__()
        self._model = m.BasicCNN3()
        self._criterion = nn.BCEWithLogitsLoss()
        self._feature = "peak"
        self._classes = 2
        self._labels = ["True", "False"]
        self._formatted_image_attribute = None
        self._formatted_prediction = None
        
    def format_image_attributes(self, image_attribute: torch.Tensor) -> None:
        self._formatted_image_attribute = image_attribute.reshape(-1, 1).float()
        
    def format_prediction(self, score: torch.tensor, threshold: float) -> None:
        self._formatted_prediction = (torch.sigmoid(score) > self.threshold).long()
        
        
        
class Photon_Energy_Configuration(Get_Configuration_Details):
    
    def __init__(self):
        super().__init__()
        self._model = m.Photon_Scattering_CNN1()
        self._criterion = nn.CrossEntropyLoss()
        self._feature = "photon_energy"
        self._classes = 3
        self._labels = [6000, 7000, 8000]
        self._formatted_image_attribute = None
        self._formatted_prediction = None
        
    def format_image_attributes(self, image_attribute: torch.Tensor) -> None:
        attribute_mapping = {6e3: 1, 7e3: 2, 8e3: 3}
        photon_energy_holder = torch.zeros_like(image_attribute, dtype=torch.long)
        for original_value, new_value in attribute_mapping.items():
            photon_energy_holder[image_attribute == original_value] = new_value
        self._formatted_image_attribute = photon_energy_holder
        
    def format_prediction(self, score: torch.tensor, threshold: float) -> None:
        _, predicted = torch.max(score, 1)
        self._formatted_prediction = predicted
        
        
class Camera_Length_Configureation(Get_Configuration_Details):
    
    def __init__(self):
        super().__init__()
        self._model = m.Camera_Length_CNN1()
        self._criterion = nn.CrossEntropyLoss()
        self._feature = "clen"
        self._classes = 3
        self._labels = [1.5, 2.5, 3.5]
        self._formatted_image_attribute = None
        self._formatted_prediction = None
        
    def format_image_attributes(self, image_attribute: torch.Tensor) -> None:
        attribute_mapping = {1.5: 1, 2.5: 2, 3.5: 3}
        camera_length_holder = torch.zeros_like(image_attribute, dtype=torch.long)
        for original_value, new_value in attribute_mapping.items():
            camera_length_holder[image_attribute == original_value] = new_value
        self._formatted_image_attribute = camera_length_holder
        
    def format_prediction(self, score: torch.tensor, threshold: float) -> None:
        _, predicted = torch.max(score, 1)
        self._formatted_prediction = predicted
