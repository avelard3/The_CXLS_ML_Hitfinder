import h5py
import numpy as np

events = 82
half_events = int(82 / 2)


with h5py.File('/scratch/sbotha/2024-hitfinder-data/epix10k2M-data/mfxly0020-r0130_294.cxi', "r") as ext_file:
        photon_energy_data = ext_file['/LCLS/photon_energy_eV'][()]
        
ext_file.close()

phot_eng_arr = np.array(photon_energy_data)
cam_len_arr = np.full(events, 0.1)

random_hit = np.ones(events)
random_hit[:half_events] = 0
np.random.shuffle(random_hit)

hit_param_arr = random_hit   

dict_3_attrs = {'camera_length': cam_len_arr, 'photon_energy' : phot_eng_arr, 'hit_parameter' : hit_param_arr}



with h5py.File("/scratch/avelard3/The_CXLS_ML_Hitfinder/src/geom_data/mfxly0020-r0130_294.cxi", "a") as f:
    for key,value in dict_3_attrs.items():
        f.attrs[key] = value

f.close()


with h5py.File("/scratch/avelard3/The_CXLS_ML_Hitfinder/src/geom_data/mfxly0020-r0130_294.cxi", "a") as fi:
    print(len(fi.attrs.keys()))
    
fi.close()