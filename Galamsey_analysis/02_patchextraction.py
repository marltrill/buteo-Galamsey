# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from glob import glob
from buteo.machine_learning.patch_extraction_v2 import extract_patches
# %%
# 10 m resolution raster files (s1 + s2), tile_size= 32
# 20 m resolution raster files (s2), tile_size= 16
# Repeat process for test sites as well
folder= "C:/Users/MALT/Desktop/Ghana/ghana_shape/"
m10 = glob(folder + "*10m*.tif")
m20 = glob(folder + "*20m*.tif")
out_path10= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/out_patches_10m/"
out_path20= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/out_patches_20m/"
training_sites= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/mines_boundaries.gpkg"
mines= "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/mines.gpkg"

extract_patches(
    m20,
    out_path20,
    tile_size=16,
    zones=training_sites,
    options=
        { "label_geom": mines },
)
# %%
#Check arrays 
import numpy as np
checkfile=  "C:/Users/MALT/Desktop/Ghana/ghana_shape/training_data_pilotarea/out_patches_10m/NDVI_10m.npy"
array= np.load(checkfile)
shape= array.shape
print(shape)
dim= array.ndim
print(dim)
size= array.size
print(size)
length= len(array)
print(length)
# %%
