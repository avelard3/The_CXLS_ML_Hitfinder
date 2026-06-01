import torch
import datetime
import os 
from . import models as m
from . import utils as u
from . import conf
import inspect
import importlib    
from torch.utils.data import DataLoader

#
class RunModel:
    
    def __init__(self, cfg: dict) -> None:
        """
        Initialize the RunModel class with model architecture, model path, output list path, and device.

        Args:
            cfg (dict): Dictionary containing important information for running (model architecture, model path, output list path, and device)
        """
        self._device = cfg['device']
        self._model_arch = cfg['model']
        self._model_path = cfg['model_path']
        self._save_output_list = cfg['save_output_list']

        self._list_containing_peaks = []
        self._list_not_containing_peaks = []
        
        self._model = None
    
    def make_model_instance(self) -> None:
        """
        This function creates an instance of the model class specified by the model architecture using a function in utils.LoadModel
        """
        self._model = u.LoadModel.make_model_instance(self._model_arch)
            
    def load_model(self) -> None:
        """
        This function loads the state dictionary into the model class to prepare it for evaluation using a function in utils.LoadModel
        """
        self._model = u.LoadModel.load_and_return_model(self._model_path, self._model, self._device)

        
    def classify_data(self, data_loader: DataLoader) -> None: 
        """
        Classify the input data using the model and segregate the data based on the classification results.
        
        Args: 
            data_loader (DataLoader):  input data in the form of a torch DataLoader object
            
        Raises:
            Exception
        """
        print('Starting classification...')
        self._model.eval()
        
        try:
            with torch.no_grad(): #? what does this mean
                for images, camera_length, photon_energy, _, paths in data_loader:
                    inputs = torch.Tensor(images).to(self._device, dtype=torch.float32)
                    cam_len = torch.Tensor(camera_length).to(self._device, dtype=torch.float32).squeeze(1)                    
                    phot_en = torch.Tensor(photon_energy).to(self._device, dtype=torch.float32).squeeze(1)                    

                    score = self._model(inputs, cam_len, phot_en)
                    prediction = (torch.sigmoid(score) > 0.5).long()
                    
                    assert len(prediction) == len(paths), "Prediction and paths length mismatch."

                    # Segregate data based on prediction
                    for pred, path in zip(prediction, paths):
                        if pred.item() == 1:
                            self._list_containing_peaks.append(path)
                            print(f'Classified as containing peaks: {path}')
                        elif pred.item() == 0:
                            self._list_not_containing_peaks.append(path)
                            print(f'Classified as not containing peaks: {path}')

        except Exception as e:
            print(f"An unexpected error occurred while classifying data: {e}")
                
    def get_classification_results(self) -> tuple:
        """
        Return the classification results as a tuple containing two lists (one with file paths with peaks and the other not) from the model.
        """
        return (self._list_containing_peaks, self._list_not_containing_peaks)
    
    def create_model_output_lst_files(self) -> None:
        """
        Create .lst files for the classified data based on the model's predictions.
        
        Raises:
            Exception while writing to list file
            Exception while creating .lst file
        """
        try:
            now = datetime.datetime.now()
            formatted_date_time = now.strftime("%m%d%y-%H%M")
            print(f'Formatted date and time: {formatted_date_time}')
            filename_peaks = f"found_peaks-{formatted_date_time}.lst"
            print(f'Filename peaks: {filename_peaks}')
            file_path_peaks = os.path.join(self._save_output_list, filename_peaks)
            print(f'File path peaks: {file_path_peaks}')
            
            filename_no_peaks = f"no_peaks-{formatted_date_time}.lst"
            file_path_no_peaks = os.path.join(self._save_output_list, filename_no_peaks)

            try:
                with open(file_path_peaks, 'w') as file:
                    for item in self._list_containing_peaks:
                        file.write(f"{item}\n")
                print(f"Created .lst file for predicted peak files. There are {len(self._list_containing_peaks)} files containing peaks.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_peaks}: {e}")

            try:
                with open(file_path_no_peaks, 'w') as file:
                    for item in self._list_not_containing_peaks:
                        file.write(f"{item}\n")
                print(f"Created .lst file for predicted empty files. There are {len(self._list_not_containing_peaks)} files without peaks.")
            except Exception as e:
                print(f"An error occurred while writing to {file_path_no_peaks}: {e}")

        except Exception as e:
            print(f"An unexpected error occurred while creating .lst files: {e}")
        
    def output_verification(self, size: int, events: int) -> None:
        #FIXME Delete this function? It's not used anywhere and isn't up-to-date?
        """
        Verify that the number of input file paths matches the sum of the output file paths by comparing the size of input file list to sum of two output file lists.
        
        Args:
            size (int): number of input images (?)
            events (int): number of images sorted (?)
        """
        if size == len(self._list_containing_peaks) // events + len(self._list_not_containing_peaks) // events:
            print("There is the same amount of input files as output files.")
        else:
            print("OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.")           
            print(f'Input H5 files: {size}\nOutput peak files: {len(self._list_containing_peaks)}\nOutput empty files: {len(self._list_not_containing_peaks)}')

        