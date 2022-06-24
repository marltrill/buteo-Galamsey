# %%

import sys, os
sys.path.append("../../")
os.environ["PROJ_LIB"] = "C:/Program Files/GDAL/projlib"
from buteo.earth_observation.s2_utils import (
    get_tiles_from_geom,
    get_tile_geom_from_name,
    unzip_files_to_folder,
    get_tile_files_from_safe_zip,
)
from buteo.earth_observation.download import download_s2_tile, download_s1_tile
from buteo.earth_observation.s2_mosaic import mosaic_tile_s2, join_s2_tiles
from buteo.earth_observation.s1_mosaic import mosaic_s1
from buteo.earth_observation.s1_mosaic import sort_rasters
from buteo.earth_observation.s1_preprocess import backscatter
from buteo.utils import delete_files_in_folder, make_dir_if_not_exists
from glob import glob
from shutil import rmtree

# %%
# Project folder
folder = "C:/Users/MALT/Desktop/Ghana/"

#Create folders if they don't exist
folder_s1_raw = make_dir_if_not_exists(folder + "s1_raw/")
folder_s1_mosaic = make_dir_if_not_exists(folder + "s1_mosaic/")
folder_s2_raw = make_dir_if_not_exists(folder + "s2_raw/")
folder_s2_mosaic = make_dir_if_not_exists(folder + "s2_mosaic/")
folder_raster = make_dir_if_not_exists(folder + "raster/")  # final folder
tmp_dir = make_dir_if_not_exists(folder + "tmp/")

#Project area in EPSG code
project_area = folder + "ghana.gpkg"
project_epsg = 32630

#Intersect Sentinel 2 tiles with project area
project_tiles = get_tiles_from_geom(project_area)

#Project dates for Sentinel-2
project_start = "20211101"  # yyyy-mm-dd
project_end = "20220201"  # yyyy-mm-dd

#Project dates for Sentinel-1
project_start_s1 = "20211206"  # yyyy-mm-dd
project_end_s1 = "20211220"  # yyyy-mm-dd

scihub_username = "your_scihub_username"
scihub_password = "your_scihubpassword"
onda_user = "your_ONDA_user"
onda_pass = "your_ONDA_password"

# Download raw sentinel 2 files

## (Optional)Only use if you need to skip files that are already downloaded
# for tile in project_tiles:
#     if tile in ["T30NVL", "T30NVM","T30NVN","T30NVP", "T30NWL", "T30NWM", "T30NWN","T30NWP", "T30NXL", "T30NXM", "T30NXN", "T30NXP", "T30NYL","T30NYM","T30NYN", "T30NYP","T30NZM", "T30NZN"]:
#         continue

#Download Sentinel-2 zipped files
for tile in project_tiles:
    download_s2_tile(
        scihub_username,
        scihub_password,
        onda_user,
        onda_pass,
        folder_s2_raw,
        tile,
        date=(project_start, project_end),
        clouds=20,
    )


#Download Sentinel-1 zipped files
for tile in project_tiles:
    download_s1_tile(
        scihub_username,
        scihub_password,
        onda_user,
        onda_pass,
        folder_s1_raw,
        get_tile_geom_from_name(tile),
        min_overlap=0.1,
        date=(project_start_s1, project_end_s1),
    )

#PROCESS SENTINEL-2
#Unzip files
    unzip_files_to_folder(
        get_tile_files_from_safe_zip(folder_s2_raw, tile),
        tmp_dir,
    )

#Create mosaics for Sentinel-2 bands
    mosaic_tile_s2(
        tmp_dir,
        tile,
        folder_s2_mosaic,
        process_bands=[
            {"size": "10m", "band": "B02"},
            {"size": "10m", "band": "B03"},
            {"size": "10m", "band": "B04"},
            {"size": "20m", "band": "B05"},
            {"size": "20m", "band": "B06"},
            {"size": "20m", "band": "B07"},
            {"size": "20m", "band": "B8A"},
            {"size": "10m", "band": "B08"},
            {"size": "20m", "band": "B11"},
            {"size": "20m", "band": "B12"},
        ],
    )

## (Optional) Remove the unzipped Sentinel-2 files
#     tmp_files = glob(tmp_dir + "*.SAFE")
#     for f in tmp_files:
#         rmtree(f)

#Merge the Sentinel 2 mosaics 10m and 20m bands seperately
join_s2_tiles(
    folder_s2_mosaic,
    folder_raster,
    folder_s2_mosaic,
    harmonisation=True,
    pixel_height=10.0,
    pixel_width=10.0,
    # nodata_value=None,
    bands_to_process=[
        "B02_10m",
        "B03_10m",
        "B04_10m",
        "B08_10m",
    ],
    clip_geom= project_area,
    projection_to_match=project_epsg,
)

join_s2_tiles(
    folder_s2_mosaic,
    folder_raster,
    folder_s2_mosaic,
    harmonisation=True,
    pixel_height=20.0,
    pixel_width=20.0,
    # nodata_value=None,
    bands_to_process=[
        "B05_20m",
        "B06_20m",
        "B07_20m",
        "B8A_20m",
        "B11_20m",
        "B12_20m",
    ],
    clip_geom= project_area,
    projection_to_match=project_epsg,
)

# #exit()
##(Optional) 
#Delete_files_in_folder(tmp_dir)

# #(Optional) If .tif files are not produced correctly.Create them from .dim and convert to dB
# from buteo.earth_observation.s1_preprocess import convert_to_tiff

# dims = glob(tmp_dir + "*step_2.dim")

# for dim in dims:
#     convert_to_tiff(dim, tmp_dir, True)

#PROCESS SENTINEL-1
#Master rasters
s2_mosaic_B12 = folder_raster + "B12_20m.tif"
s2_mosaic_B04 = folder_raster + "B04_10m.tif"

#Unzip files
zip_files_s1 = glob(folder_s1_raw + "*.zip")
for idx, image in enumerate(zip_files_s1):
    name = os.path.splitext(os.path.basename(image))[0]
    name = name.split(".")[0]

    if len(glob(tmp_dir + name + "*.*")) > 0:
        print(f"{name} already processed, skipping..")
        continue

    try:
        backscatter(
            image,
            tmp_dir,
            tmp_dir,
            extent=s2_mosaic_B12,
            epsg=project_epsg,
            decibel=True,
        )
    except Exception as e:
        raise Exception(f"error with image: {image}, {e}")

    print(f"completed {idx+1}/{len(zip_files_s1)}")
 
vv_paths= glob(tmp_dir + "*_Gamma0_VV.tif")

mosaic_s1(
    vv_paths,
    folder_raster + "VV_10m_.tif",
    tmp_dir,
    s2_mosaic_B04,
    skip_completed= True,
)
print("VV_10m raster completed")

vh_paths= glob(tmp_dir + "*_Gamma0_VH.tif")
mosaic_s1(
    vh_paths,
    folder_raster + "VH_10m_.tif",
    tmp_dir,
    s2_mosaic_B04,
    skip_completed= True,
)
print("VH_10m raster completed")

# ##(Optional) I ran out of memory, so I had to split my project area in several segments.

# folder_in= "C:/Users/MALT/Desktop/Gamma0/"
# s2_1 = folder_raster + "B04_10m_1.tif"
# s2_1A = folder_raster + "B04_10m_1A.tif"
# s2_1B = folder_raster + "B04_10m_1B.tif"
# s2_2 = folder_raster + "B04_10m_2.tif"
# s2_2A = folder_raster + "B04_10m_2A.tif"
# s2_2A_A = folder_raster + "B04_10m_2A_A.tif"
# s2_2A_B = folder_raster + "B04_10m_2A_B.tif"
# s2_2B = folder_raster + "B04_10m_2B.tif"
# s2_3 = folder_raster + "B04_10m_3.tif"
# s2_3A = folder_raster + "B04_10m_3A.tif"
# s2_3A_A = folder_raster + "B04_10m_3A_A.tif"
# s2_3A_B = folder_raster + "B04_10m_3A_B.tif"
# s2_3B = folder_raster + "B04_10m_3B.tif"
# s2_4 = folder_raster + "B04_10m_4.tif"
# s2_4A = folder_raster + "B04_10m_4A.tif"
# s2_4B = folder_raster + "B04_10m_4B.tif"
# s2_4C = folder_raster + "B04_10m_4C.tif"

# out_dir = "C:/Users/MALT/Desktop/Ghana/SAR1/"

# vv_paths = sort_rasters(glob(tmp_dir + "*_Gamma0_VV.tif"))
# vh_paths = sort_rasters(glob(tmp_dir + "*_Gamma0_VH.tif"))

# folder_s1_mosaic2 = "C:/Users/MALT/Desktop/ICZM_sentinel/S1_mosaic2/"
# vv_paths2 = sort_rasters(glob(folder_s1_mosaic2 + "*_Gamma0_VV.tif"))
# vh_paths2 = sort_rasters(glob(folder_s1_mosaic2 + "*_Gamma0_VH.tif"))

# folder_s1_mosaic3 = "C:/Users/MALT/Desktop/ICZM_sentinel/S1_mosaic3/"
# vv_paths3 = sort_rasters(glob(folder_s1_mosaic3 + "*_Gamma0_VV.tif"))
# vh_paths3 = sort_rasters(glob(folder_s1_mosaic3 + "*_Gamma0_VH.tif"))

# folder_s1_mosaic4 = "C:/Users/MALT/Desktop/ICZM_sentinel/S1_mosaic4/"
# vv_paths4 = sort_rasters(glob(folder_s1_mosaic4 + "*_Gamma0_VV.tif"))
# vh_paths4 = sort_rasters(glob(folder_s1_mosaic4 + "*_Gamma0_VH.tif"))

# #Mosaic the sentinel 1 images

# # PART 1
# mosaic_s1(
#     vv_paths,
#     folder_raster + "VV_10m_2B_3.tif",
#     tmp_dir,
#     s2_2B,
#     chunks=4,
#     skip_completed=True,
# )
# print("VV_10m_2B raster completed")

# mosaic_s1(
#     vh_paths,
#     folder_raster + "VH_10m_3B.tif",
#     tmp_dir,
#     s2_3B,
#     chunks=4,
#     skip_completed=False,
# )
# print("VH_10m_3B raster completed")

# # PART 2
# mosaic_s1(
#     vv_paths,
#     folder_raster + "VV_10m_2A_A_unint8.tif",
#     tmp_dir,
#     s2_2A_A,
#     chunks=4,
#     skip_completed=False,
# )
# print("VV_10m_2A_A_unint8 raster completed")

# mosaic_s1(
#     vh_paths,
#     folder_raster + "VH_10m_2.tif",
#     tmp_dir,
#     s2_2,
#     chunks=4,
#     skip_completed=False,
# )
# print("VH_10m_2 raster completed")

# # PART 3
# mosaic_s1(
#     vv_paths,
#     folder_raster + "VV_10m_3.tif",
#     tmp_dir,
#     s2_3,
#     chunks=4,
#     skip_completed=False,
# )
# print("VV_10m_3 raster completed")

# mosaic_s1(
#     vh_paths,
#     folder_raster + "VH_10m_3.tif",
#     tmp_dir,
#     s2_3,
#     chunks=4,
#     skip_completed=False,
# )
# print("VH_10m_3 raster completed")

# # PART 4
# mosaic_s1(
#     vv_paths,
#     folder_raster + "VV_10m_4.tif",
#     tmp_dir,
#     s2_4,
#     chunks=4,
#     skip_completed=False,
# )
# print("VV_10m_4 raster completed")

# mosaic_s1(
#     vh_paths,
#     folder_raster + "VH_10m_4.tif",
#     tmp_dir,
#     s2_4,
#     chunks=4,
#     skip_completed=False,
# )
# print("VH_10m_4 raster completed")

# mosaic_s1(
#     vv_paths2,
#     out_dir + "VV_10m_2.tif",
#     tmp_dir,
#     s2_2,
#     chunks=4,
#     skip_completed=False,
# )

# mosaic_s1(
#     vh_paths2,
#     out_dir + "VH_10m_2.tif",
#     tmp_dir,
#     s2_2,
#     chunks=4,
#     skip_completed=False,
# )

# mosaic_s1(
#     vv_paths3,
#     out_dir + "VV_10m_3.tif",
#     tmp_dir,
#     s2_3,
#     chunks=4,
#     skip_completed=False,
# )

# mosaic_s1(
#     vh_paths3,
#     out_dir + "VH_10m_3.tif",
#     tmp_dir,
#     s2_3,
#     chunks=4,
#     skip_completed=False,
# )

# mosaic_s1(
#     vv_paths4,
#     out_dir + "VV_10m_4.tif",
#     tmp_dir,
#     s2_4,
#     chunks=4,
#     skip_completed=False,
# )

# mosaic_s1(
#     vh_paths4,
#     out_dir + "VH_10m_4.tif",
#     tmp_dir,
#     s2_4,
#     chunks=4,
#     skip_completed=False,
# )

# %%
