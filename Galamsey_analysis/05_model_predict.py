# %%
import sys, os, gc
sys.path.append("../../")
sys.path.append("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin/")

import numpy as np
from glob import glob
from osgeo import gdal
from buteo.machine_learning.tensorflow_utils import tpe
from tensorflow.python.keras.models import load_model
from buteo.raster.io import stack_rasters, raster_to_array
from buteo.machine_learning.patch_extraction_v2 import predict_raster
from buteo.raster.io import (
    raster_to_array,
    array_to_raster,
    stack_rasters_vrt,
    )
from buteo.raster.clip import internal_clip_raster, clip_raster
from buteo.machine_learning.ml_utils import (
    preprocess_optical,
    preprocess_sar,
    get_offsets,
  )
# %%
#Use eGPU
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

raster_folder = "D:/MALT/ghana/"
model_folder =  "D:/MALT/ghana/models/VI/2/"
outdir = "D:/MALT/ghana/predictions_VI/v2_vi03/"

convert_db = False
test = False
batch_size_predict = 256
nodata_value = 0

model_path = model_folder + f"ghana_mines_v2_vi03"
out_mosaic = outdir + f"ghana_mines_v2_VI03.tif"
out_mosaic_rounded = outdir + f"ghana2_mines_v2_VI03.tif"

print("Loading model.")
loaded_model = load_model(model_path, custom_objects={"tpe": tpe})

for region in glob(raster_folder + "grid_test/*.gpkg"):
    region_name = os.path.splitext(os.path.basename(region))[0]
    outname = region_name
    gc.collect()

    if test and region_name != "fid_1":
        continue

    if not test and region_name in ["fid_2", "fid_1"]:
        continue

    if os.path.exists(outdir + f"{outname}.tif"):
        continue

    print(f"Processing region: {region_name}")

    print("Clipping RESWIR.")
    b20m_clip = internal_clip_raster(
        raster_folder + f"B05_20m.tif",
        region,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/20m_clip.tif",
    )

    reswir = clip_raster(
        [
            raster_folder + "B05_20m.tif",
            raster_folder + "B06_20m.tif",
            raster_folder + "B07_20m.tif",
            raster_folder + "B11_20m.tif",
            raster_folder + "B12_20m.tif",
        ],
        clip_geom=region,
        adjust_bbox=False,
        all_touch=False,
        dst_nodata=nodata_value,
    )

    print("Stacking RESWIR.")
    reswir_stack = []
    for idx, raster in enumerate(reswir):
        reswir_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(reswir[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=10000,
                ),
                reference=reswir[idx],
            ),
        )
    reswir_stacked = stack_rasters(reswir_stack, dtype="float32")
    for raster in reswir:
        gdal.Unlink(raster)

    print("Clipping RGBN and VEG.")
    b10m_clip = internal_clip_raster(
        raster_folder + "B04_10m.tif",
        b20m_clip,
        adjust_bbox=False,
        all_touch=False,
        out_path="/vsimem/10m_clip.tif",
    )
    rgbn = clip_raster(
        [
            raster_folder + "B02_10m.tif",
            raster_folder + "B03_10m.tif",
            raster_folder + "B04_10m.tif",
            raster_folder + "B08_10m.tif",
            raster_folder + "NDVI_10m.tif",
            raster_folder + "NDMI_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
        dst_nodata=nodata_value,
    )

    print("Stacking RGBN.")
    rgbn_stack = []
    for idx, raster in enumerate(rgbn):
        rgbn_stack.append(
            array_to_raster(
                preprocess_optical(
                    raster_to_array(rgbn[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_high=10000,
                ),
                reference=rgbn[idx],
            ),
        )
    rgbn_stacked = stack_rasters(rgbn_stack, dtype="float32")
    for raster in rgbn:
        gdal.Unlink(raster)

    print("Clipping SAR.")
    sar = clip_raster(
        [
            raster_folder + "VV_10m.tif",
            raster_folder + "VH_10m.tif",
        ],
        clip_geom=b20m_clip,
        adjust_bbox=False,
        all_touch=False,
        dst_nodata=nodata_value,
    )

    print("Stacking SAR.")
    sar_stack = []
    for idx, raster in enumerate(sar):
        sar_stack.append(
            array_to_raster(
                preprocess_sar(
                    raster_to_array(sar[idx]),
                    target_low=0,
                    target_high=1,
                    cutoff_low=-30,
                    cutoff_high=20,
                    convert_db=convert_db,
                ),
                reference=sar[idx],
            ),
        )
    sar_stacked = stack_rasters(sar_stack, dtype="float32")
    for raster in sar:
        gdal.Unlink(raster)

    print("Ready for predictions.")

    predict_raster(
        [rgbn_stacked, sar_stacked, reswir_stacked],
        tile_size=[32, 32, 16],
        output_tile_size=32,
        model_path=model_path,
        loaded_model=loaded_model,
        reference_raster=b10m_clip,
        out_path=outdir + f"{outname}.tif",
        offsets=[
            get_offsets(32),
            get_offsets(32),
            get_offsets(16),
        ],
        batch_size=batch_size_predict,
        output_channels=1,
        scale_to_sum=False,
        method="mad",
    )

    try:
        for raster in reswir_stack:
            gdal.Unlink(raster)

        for raster in rgbn_stack:
            gdal.Unlink(raster)

        for raster in sar_stack:
            gdal.Unlink(raster)

        gdal.Unlink(reswir_stacked)
        gdal.Unlink(rgbn_stacked)
        gdal.Unlink(sar_stacked)
        gdal.Unlink(b10m_clip)
    except:
        pass

if test:
    exit() #uncomment if I'm predicting on a particular extent, not a mosaic

print("Creating prediction mosaic.") #predicting for all s2 tiles (the pilot area split into 9)
mosaic = stack_rasters_vrt(
    glob(outdir + "/id_*.tif"),
    "/vsimem/vrt_predictions.vrt",
    seperate=False,
)
mosaic = "/vsimem/vrt_predictions.vrt"

internal_clip_raster(
    mosaic,
    raster_folder + "ghana.gpkg",
    out_path=out_mosaic,
    postfix="",
)

rounded = array_to_raster(
    np.clip(np.rint(raster_to_array(mosaic)), 0, 100).astype("uint8"), mosaic
)

internal_clip_raster(
    rounded,
    raster_folder + "ghana.gpkg",
    out_path=out_mosaic_rounded,
    postfix="",
)
