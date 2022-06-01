# Buteo - Facilitating EO-Driven Decision Support Systems

The Buteo-Toolbox is a series of modules that ease the creation of Earth Observation Driven Spatial Decision Support Systems. The modules are located in the lib folder, geometry used for testing and clipping is located in geometry. In the examples folder there are jupyter notebooks that showcase analysis' done using the toolbox.

There are modules are:

## raster

- read and verify a raster or a list of rasters
- clip rasters to other rasters or vectors
- align a list of rasters for analysis
- shift, resample and reproject raster data
- easily manage nodata values
- parallel zonal statistics (link - host)

## vector

- read and verify integrity
- parallel zonal statistics (link)
- clip, buffer

## filter

- custom convolution based filters.
- global filters and multispectral indices (link - host)
- textures for analysis
- noise reduction of SAR imagery (link - host)
- kernel designer & reduction bootcamp

## terrain

- download srtm, aster and the danish national DEM
- basic propocessing of the DEM's.

## earth_observation

- download sentinel 1, 2, 3, 5, landsat and modis data
- process sentinel 1 and 2 (sentinel 1 requires esa-snap dep.)
- generate mosaics of sentinel 1 and 2
- pansharpen bands
- noise reduction of SAR imagery (link)
- multispectral indices (link)

## machine_learning

- patch extraction of tiles and geometries, allows overlaps, for CNN's
- machine learning utilities library: rotate images, add noise etc..
- model for building extraction for sentinel 1 and 2
- model for pansharpening sentinel 2
- model for noise reduction sentinel 1

## extra

- custom orfeo-toolbox python bindings
- ESA snap GPT python bindings and graphs

The system is under active development and is not ready for public release. It is being developed by NIRAS and Aalborg University.

# Dependencies

numpy
numba
pandas
sentinelsat
tqdm
gdal # install on your own

optional:
orfeo-toolbox
esa-snap

# Todo

- finish filters library - kernel tests
- update zonal statistics & parallelise vector math
- remove dependencies: sen1mosaic
- create models for pansharpening, buildings and noise-reduction
- generate examples
- synthetic sentinel 2?

# Functions todo

raster_footprints
raster_mosaic
raster_proximity
raster_hydrology
raster_vectorize

vector_grid
vector_select
vector_buffer_etc..

machine_learning_extract_sample_points
