# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

import numpy as np
from osgeo import gdal
from glob import glob
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
)
# %%
#VEGETATION INDICES (included in RGBN stack)
#Repeat process for test sites as well
#Step 1: this step normalized the values between 0 and 1
#Project folders
folder_10m= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/out_patches_10m/"
folder_20m= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/out_patches_20m/"
out_path= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/"

def preprocess(
    folder_10m,
    folder_20m,
    out_path,
    low=0,
    high=1,
    optical_top=8000,
):
    b02 = folder_10m + "B02_10m.npy"
    b03 = folder_10m + "B03_10m.npy"
    b04 = folder_10m + "B04_10m.npy"
    b08 = folder_10m + "B08_10m.npy"

    b05 = folder_20m + "B05_20m.npy"
    b06 = folder_20m + "B06_20m.npy"
    b07 = folder_20m + "B07_20m.npy"
    b11 = folder_20m + "B11_20m.npy"
    b12 = folder_20m + "B12_20m.npy"

    vv = folder_10m + "VV_10m.npy"
    vh = folder_10m + "VH_10m.npy"

    NDVI = folder_10m + "NDVI_10m.npy"
    NDMI = folder_10m + "NDMI_10m.npy"

    target = "area"

    label_area = folder_10m + f"label_{target}_10m.npy"

    area = np.load(label_area)
    shuffle_mask = np.random.permutation(area.shape[0])

    label_out = out_path + f"label_{target}_10m.npy"

    np.save(label_out, area[shuffle_mask])

    print ("Preprocessing RGBN and VI") 
    rgbn = preprocess_optical(
        np.stack(
            [
                np.load(b02),
                np.load(b03),
                np.load(b04),
                np.load(b08),
                np.load(NDVI),
                np.load(NDMI),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(out_path + "RGBN.npy", rgbn[shuffle_mask])
    
    print ("Preprocessing RESWIR") 
    reswir = preprocess_optical(
        np.stack(
            [
                np.load(b05),
                np.load(b06),
                np.load(b07),
                np.load(b11),
                np.load(b12),
            ],
            axis=3,
        )[:, :, :, :, 0],
        target_low=low,
        target_high=high,
        cutoff_high=optical_top,
    )

    np.save(out_path + "RESWIR.npy", reswir[shuffle_mask])

    print ("Preprocessing SAR") 
    sar_stacked = np.stack(
        [
            np.load(vv),
            np.load(vh),
        ],
        axis=3,
    )[:, :, :, :, 0]
    sar = preprocess_sar(
        sar_stacked,
        target_low=low,
        target_high=high,
        convert_db=False, #my s1 rasters are already converted to dB
    )

    np.save(out_path + "SAR.npy", sar[shuffle_mask])

preprocess(folder_10m, folder_20m, out_path)
# %%
# Step 2: Save normalisation arrays (.npy) to .npz (compressed file)
# Note that RGBN contains NDVI and NDMI

outfile= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/ghana_mines_train.npz"
SAR=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/SAR.npy"
sar= np.load(SAR)
RESWIR=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/RESWIR.npy"
reswir= np.load(RESWIR)
RGBN=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/RGBN.npy"
rgbn= np.load(RGBN)
LABELS=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/label_area_10m.npy"
labels= np.load(LABELS)
np.savez(outfile, sar=sar, reswir=reswir, rgbn=rgbn, labels=labels)

# %%
#Step 3: Check .npz compressed file
import numpy as np
checkfile=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/normalisation/ghana_mines_train.npz"
array= np.load(checkfile)
array_files= array.files
print (array.files)
#Shape
sar_a= array['sar'].shape
print(sar_a)
# It is 32 by 32, 2 bands (s1 10m)

rgbn_a= array['rgbn'].shape
print(rgbn_a)
#It is 32 by 32, 6 bands (s2 10m and vegetation indices)

reswir_a= array['reswir'].shape
print(reswir_a)
#It is 16 by 16, 5 bands (s2 20m)

labels_a= array['labels'].shape
print(labels_a)
#It is 32 by 32, 1 band

#Check min,max values. They should be between 0 and 1. Repeat for all
print (array['sar'].min())
print (array['sar'].max())

#Check min,max values. They should be between 0 and 1
print (array['rgbn'].min())
print (array['rgbn'].max())

#Check min,max values. They should be between 0 and 1
print (array['reswir'].min())
print (array['reswir'].max())
