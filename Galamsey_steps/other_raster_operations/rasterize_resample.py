# %%
import sys, os

sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"

from osgeo import gdal
from glob import glob
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
from buteo.vector.rasterize import rasterize_vector
from buteo.raster.resample import internal_resample_raster
from buteo.machine_learning.patch_extraction import extract_patches
# %%
# STEP 1
# MAKE SURE TO ADD A FIELD IN THE VECTOR FILE WITH A CLASS TO BURN, EXAMPLE= 1 (in all rows)
import time
start = time.time()
extent= "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/testsite3.gpkg"
mines = "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/mines3.gpkg"
mines_rasterized= "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/mines3.tif"
print("rasterizing vector.")

#Make sure that the "Class" field supports the data type, in this case dtype="uint8"
rasterize_vector(
    vector= mines,
    pixel_size= 0.5, #Rastertize vector to 50cm
    out_path=mines_rasterized,
    extent=extent,
    all_touch=False,
    dtype="uint8",
    optim="raster",
    band=1,
    fill_value=0,
    nodata_value=None,
    check_memory=True,
    burn_value=1,
    attribute="burn",
    )
print("rasterizing vector finished.")
# # %%
# print("writing rasterized 50cm final output.")
# array_to_raster(
#         (raster_to_array(buildings_rasterized)).astype(
#             "uint8"
#         ),
#         reference=buildings_rasterized,
#         # out_path=folder + f"fid_{number}_rasterized.tif",
#         #out_path=buildings_rasterized_out,
#     )
# end = time.time()
# print(end - start)
# %%
# STEP 2: Resample rasters to match the 10m by 10m predictions
mines_rasterized= "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/mines3.tif"
mines_resampled= "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/mines3_10m.tif"
mines_rr= "C:/Users/MALT/Desktop/Ghana/predictions_ghana_v8_03/accuracy_2/mines3_10m_F.tif"

print("resampling.")
internal_resample_raster(
        mines_rasterized,
        10.0,
        resample_alg="average",
        out_path= mines_resampled,
    )
print("writing final output.")
array_to_raster(
        (raster_to_array(mines_resampled)*100).astype(
            "float32"
        ),
        reference=mines_resampled,
        out_path=mines_rr,
    )

gdal.Unlink(mines_rasterized)
gdal.Unlink(mines_resampled)
# %%