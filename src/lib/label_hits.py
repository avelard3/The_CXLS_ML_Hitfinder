import h5py as h5
import numpy  as np

with h5.File(image_name, 'w') as f:
    print("Make sure you gave me the good_images")
    dataset = f.create_dataset('/hit/', shape=image_size, dtype=int)
    hit_val_array = []
    for i  in image_size:
        if num=exists:
            hit_val_array = np.append(1)
        else:
            hit_val_array = np.append(0)
    
    dataset = hit_val_array
    
print(f"A new file was created in {user_location} called {image_name}")