
# %%
import sys, os
sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"
from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.convolutions import filter_array
# %%

folder = "C:/Users/MALT/Desktop/Ghana/predictions/"
in_raster = folder + "ghana_mines_v2_VI03.tif"
out_raster = folder + "ghana_mines_v2_VI03_med3_o3_c5.tif"
in_raster_array = raster_to_array(in_raster)

filter_median = 3
filter_size_open = 3
filter_size_close = 5

# median
# (filter_median, filter_median, 1)=  ((shape:3,3), (sigma: 1))= (3,3,1)
in_raster_median = filter_array(in_raster_array, (filter_median, filter_median, 1), distance_calc=False, operation="median")

# open
in_raster_erode = filter_array(in_raster_median, (filter_size_open, filter_size_open, 1), operation="erode", distance_calc=False, normalised=False)
in_raster_dilate = filter_array(in_raster_erode, (filter_size_open, filter_size_open, 1), operation="dilate", distance_calc=False, normalised=False)

# close
in_raster_dilate2 = filter_array(in_raster_dilate, (filter_size_close, filter_size_close, 1), operation="dilate", distance_calc=False, normalised=False)
in_raster_erode2 = filter_array(in_raster_dilate2, (filter_size_close, filter_size_close, 1), operation="erode", distance_calc=False, normalised=False)

array_to_raster(in_raster_erode2, reference=in_raster, out_path=out_raster)
# %%
