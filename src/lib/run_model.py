import torch
import datetime
import os 
from . import models as m
from . import utils as u
from . import conf
import inspect
import importlib    


class RunModel:
    
    def __init__(self, cfg: dict, model_inputs: dict) -> None:
        """
        Initialize the RunModel class with model architecture, model path, output list path, h5 file paths, and device.

        Args:
            cfg (dict): Dictionary containing important information for training including: data loaders, batch size, training device, number of epochs, the optimizer, the scheduler, the criterion, the learning rate, and the model class. Everything besides the data loaders and device are arguments in the sbatch script.
            attributes (dict): Dictionary containing the names of the metadata contained in the h5 image files. These names could change depending on whom created the metadata, so the specific names are arguments in the sbatch script. 
        """
        self._device = cfg['device']
        self._model_arch = cfg['model']
        self._model_path = cfg['model_path']
        self._save_output_list = cfg['save_output_list']

        self._model_in = model_inputs
        self._list_containing_peaks = []
        self._list_not_containing_peaks = []
        
        self._model = None
    
    def make_model_instance(self) -> None:
        """
        Create an instance of the model class specified by the model architecture.
        """
        try:
            self._model = getattr(m, self._model_arch)(model_inputs=self._model_in)
            print(f'Model object has been created: {self._model.__class__.__name__}')
        except AttributeError:
            print(f"Error: Model '{self._model_arch}' not found in the module.")
            print(f'Available models: {inspect.getmembers(m, inspect.isclass)}')
        except TypeError:
            print(f"Error: '{self._model_arch}' found in module is not callable.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            
    def load_model(self) -> None:
        """
        Load the state dictionary into the model class and prepare it for evaluation.
        """
        self._model = u.LoadModel.load_and_return_model(self._model_path, self._model, self._device)
        
        # try:
        #     state_dict = torch.load(self._model_path)
        #     self._model.load_state_dict(state_dict)
        #     #self._model.eval() 
        #     self._model.to(self._device)
        #     print(f'The model state dict has been loaded into: {self._model.__class__.__name__}')
            
        # except FileNotFoundError:
        #     print(f"Error: The file '{self.transfer_learning_path}' was not found.")
        # except torch.serialization.pickle.UnpicklingError:
        #     print(f"Error: The file '{self.transfer_learning_path}' is not a valid PyTorch model file.")
        # except RuntimeError as e:
        #     print(f"Error: There was an issue loading the state dictionary into the model: {e}")
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")
        
    def classify_data(self, data_loader) -> None:
        """
        Classify the input data using the model and segregate the data based on the classification results.
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
        """
        Verify that the number of input file paths matches the sum of the output file paths by comparing the size of input file list to sum of two output file lists.
        """
        if size == len(self._list_containing_peaks) // events + len(self._list_not_containing_peaks) // events:
            print("There is the same amount of input files as output files.")
        else:
            print("OUTPUT VERIFICATION FAILED: The input paths do not match the output paths.")           
            print(f'Input H5 files: {size}\nOutput peak files: {len(self._list_containing_peaks)}\nOutput empty files: {len(self._list_not_containing_peaks)}')
