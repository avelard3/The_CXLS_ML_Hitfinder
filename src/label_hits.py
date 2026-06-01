import h5py as h5
import numpy  as np
import argparse
import os
import re

def arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: 
    """
    This function is for adding an argument when running the python file. 
    It needs to take an lst file of the h5 files for the model use. 
    
    Args:
        parser (argparse.ArgumentParser): The argument parser to which the arguments will be added.
        
    Returns:
        argparse.ArgumentParser: The parser with the added arugments.
    """
    parser.add_argument('-l', '--list', type=str, help='File path to the .lst file containing file paths to the .h5 file to run through the model.')
    parser.add_argument('-sl', '--save_location', type=str, help="Place where hit file should be saved")
    parser.add_argument('-ni', '--num_images_per_file', type=int, help="Number of images per file")
        
    try:
        args = parser.parse_args()
        print("Parsed arguments:")
        for arg, value in vars(args).items():
            print(f"{arg}: {value}")
            
        return args
    
    except argparse.ArgumentError as e:
        print(f"Argument error: {e}")
    
    except argparse.ArgumentTypeError as e:
        print(f"Argument type error: {e}")
    
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def create_hit_dataset(image_name, user_location, num_img):
    og_filename = image_name.strip()
    og_filename = os.path.basename(og_filename)
    clean_filename = og_filename.replace("goodEvents-advanceSort-", "")
    clean_filename = clean_filename.replace(".list", ".h5")
    full_path = f'{user_location}{clean_filename}' 

    
    hit_val_array = np.array([])
        
    with open(image_name.strip(), "r") as f:
        lines = f.readlines()
        
    j=0
    for i in range(num_img):
        
        line = lines[j].strip()
        if re.search(rf"//\b{i}\b", line):
            hit_val_array = np.append(hit_val_array, 1.0)
            if j < (len(lines)-1):
                j+=1            
        else:
            hit_val_array = np.append(hit_val_array, 0.0)

    print("hit_val_array", hit_val_array)

    with h5.File(full_path, 'w') as f:
        dset = f.create_dataset("/hits", data=hit_val_array)

        
    print(f"A new file was created in {full_path}")
    
def main():
    parser = argparse.ArgumentParser(description='Parameters for running a model.')
    args = arguments(parser)
    file_list = args.list
    save_location = args.save_location
    num_img_per_file = args.num_images_per_file
    
    with open(file_list, 'r') as lst_file:
        for i, source_file in enumerate(lst_file):
            print("THE SOURCE FILE IS", source_file)
            create_hit_dataset(source_file, "/scratch/avelard3/NSLS-2019-August/h5_hits/", 500)
    

    
if __name__ == '__main__':
    main()