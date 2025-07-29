import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import datetime
from torch.utils.data import DataLoader
import optuna
from . import models as m


from . import conf

class EvaluateModel:
    
    def __init__(self, cfg: dict, trained_model: nn.Module, testing_data: DataLoader) -> None:
        """
        Breaks out important dictonaries, takes in the trained model, creates a logger object, and creates parameters to store evaluation metrics. 

        Args:
            cfg (dict): Dictionary containing important information for training including 
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files
            trained_model (nn.Module): Trained model taken from the training class. 
            testing_data (DataLoader): Data that was set aside for testing/evaluating hitfinder
        """
        
        self._test_loader = testing_data
        self._device = cfg['device']
        self._model = trained_model

        
        self._confusion_matrix = None
        self._all_labels = []
        self._all_predictions = []
        self._classification_report_dict = {}


    def run_testing_set(self) -> None:
        """ 
        Runs the trained model in evaluation mode, by creating arrays of labels and predictions to compare against each other for metrics. 
        """
        print(f'Running evaluation on model: {self._model.__class__.__name__}')
        self._model.eval()

        try:
            with torch.no_grad():
                for images, camera_length, photon_energy, hit_parameter, _ in self._test_loader:
                    inputs = torch.Tensor(images).to(self._device, dtype=torch.float32)
                    cam_len = torch.Tensor(camera_length).to(self._device, dtype=torch.float32).squeeze(1)                    
                    phot_en = torch.Tensor(photon_energy).to(self._device, dtype=torch.float32).squeeze(1)                    

                    score = self._model(inputs, cam_len, phot_en)
                    truth = hit_parameter.reshape(-1, 1).float().to(self._device)
                                    
                    predictions = (torch.sigmoid(score) > 0.5).long()
                    self._all_labels.extend(torch.flatten(truth.cpu()))
                    self._all_predictions.extend(torch.flatten(predictions.cpu()))
                    
            # No need to reshape - arrays should already be flat
            self._all_labels = np.array(self._all_labels)
            self._all_predictions = np.array(self._all_predictions)
        except RuntimeError as e:
            print(f"RuntimeError during evaluation: {e}")  
        except AttributeError as e:
            print(f"AttributeError during evaluation: {e}")
        except TypeError as e:
            print(f"TypeError during evaluation: {e}")    
        except Exception as e:
            print(f"An unexpected error occurred during evaluation: {e}")

        
    def make_classification_report(self) -> None:
        """
        Creates a classification report for the model and prints it.
        """
        try:
            print('Creating classification report...')
            self._classification_report_dict = classification_report(self._all_labels, self._all_predictions, output_dict=True)
            print('Classification Matrix: ')
            [print(f"{key}: {value}") for key, value in self._classification_report_dict.items()]
        except Exception as e:
            print(f"An error occurred while creating the clasification report: {e}")       
        
    def get_classification_report(self) -> dict:
        """
        Returns the classification report for the model.
        """
        return self._classification_report_dict

        
    def plot_confusion_matrix(self, path:str = None) -> None:
        """ 
        Plots the confusion matrix of the testing set.
        The values in this matrix are done so that the rows total to 1. 
        
        Args:
            path (str): Path to where confusion matrix should be saved
             
        """
        
        try:
            print('Creating confusion matrix...')
            self._confusion_matrix = confusion_matrix(self._all_labels, self._all_predictions, normalize='true')
        except Exception as e:
            print(f"An error occurred while creating the confusion matrix: {e}")      
             
        # Plotting the confusion matrix
        try:
            plt.matshow(self._confusion_matrix, cmap="Blues")
            plt.title(f'CM for {self._model.__class__.__name__}')
            plt.colorbar()
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'confusion_matrix.png'
                plt.savefig(path)
                
                print(f'Confustion matrix saved to: {path}')
        except Exception as e:
            print(f"An error occurred while plotting confusion matrix: {e}")

    def plot_roc_curve(self, path:str=None) -> None:
        """
        Plots the ROC (Reciever Operating Characteristic) Curve of the testing set.
        The x-axis is the false positive rate and the y-axis is the true positive rate
        Args:
            path (str): Path to where ROC curve should be saved
        """
        try:
            print('Creating ROC curve...')
            roc_display = RocCurveDisplay.from_predictions(self._all_labels, self._all_predictions)
            _ = roc_display.ax_.set(
                xlabel="False Positive Rate",
                ylabel="True Positive Rate",
                title="ROC Curve",
            )
            
            if path != None:
                now = datetime.datetime.now()
                formatted_date_time = now.strftime("%m%d%y-%H%M")
                path = path + '/' + formatted_date_time + '-' + 'roc_curve.png'
                plt.savefig(path)
                
                print(f'ROC curve saved to: {path}')
        except Exception as e:
            print(f"An error occurred while creating the ROC curve: {e}")       
        
    def get_confusion_matrix(self) -> np.ndarray:
        """ 
        Returns the confusion matrix of the testing set as a numpy array.
        """
        return self._confusion_matrix