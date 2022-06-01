
# %%
import sys, os
sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"
from buteo.raster.resample import resample_raster
import glob

# %% 
folder = "C:/Users/MALT/Desktop/Ghana/sentinel_projectarea/"
images_B06 = folder + "B06_20m_compressed.tif"
images_B07 = folder + "B07_20m_compressed.tif"
images_8A = folder + "B8A_20m_compressed.tif"
images_B11 = folder + "B11_20m_compressed.tif"
images_B12 = folder + "B12_20m_compressed.tif"
images = glob(folder + "*_compressed*.tif")
out = "C:/Users/MALT/Desktop/Ghana/sentinel_projectarea/resampled/"

resample_raster(
    raster= images_B06,
    target_size=10,
    out_path=out,
    resample_alg= 'average',
    dtype='UInt16',
)
print ("B06 completed")
resample_raster(
    raster= images_B07,
    target_size=10,
    out_path=out,
    resample_alg= 'average',
    dtype='UInt16',
)
print ("B07 completed")
resample_raster(
    raster= images_8A,
    target_size=10,
    out_path=out,
    resample_alg= 'average',
    dtype='UInt16',
)
print ("B8a completed")
resample_raster(
    raster= images_B11,
    target_size=10,
    out_path=out,
    resample_alg= 'average',
    dtype='UInt16',
)
print ("B11 completed")
resample_raster(
    raster= images_B12,
    target_size=10,
    out_path=out,
    resample_alg= 'average',
    dtype='UInt16',
)
print ("B12 completed")
# %%
