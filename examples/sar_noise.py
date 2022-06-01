import sys
import numpy as np
from numba import jit, prange

sys.path.append("../")

from buteo.raster.io import raster_to_array, array_to_raster
from buteo.filters.kernel_generator import create_kernel
from buteo.filters.pansharpen import pansharpen_filter


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_quantile(values, weights, quant):
    sort_mask = np.argsort(values)
    sorted_data = values[sort_mask]
    sorted_weights = weights[sort_mask]
    cumsum = np.cumsum(sorted_weights)
    intersect = (cumsum - 0.5 * sorted_weights) / cumsum[-1]
    return np.interp(quant, intersect, sorted_data)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, inline="always")
def hood_mad(values, weights):
    median = hood_quantile(values, weights, 0.5)
    absdeviation = np.abs(np.subtract(values, median))
    return hood_quantile(absdeviation, weights, 0.5)


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def mad_match(
    arr_change,
    arr_base,
    offsets,
    weights,
    quantile=0.5,
    nodata=True,
    nodata_value=-9999.0,
):
    x_adj = arr_base.shape[0] - 1
    y_adj = arr_base.shape[1] - 1
    z_adj = (arr_base.shape[2] - 1) // 2

    hood_size = len(offsets)
    if nodata:
        result = np.full(arr_base.shape[:2], nodata_value, dtype="float32")
    else:
        result = np.zeros(arr_base.shape[:2], dtype="float32")

    for x in prange(arr_base.shape[0]):
        for y in range(arr_base.shape[1]):

            hood_values_change = np.zeros(hood_size, dtype="float32")
            hood_values_base = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]
                offset_z = offsets[n][2]

                outside = False

                if offset_z < -z_adj:
                    offset_z = -z_adj
                    outside = True
                elif offset_z > z_adj:
                    offset_z = z_adj
                    outside = True

                if offset_x < 0:
                    offset_x = 0
                    outside = True
                elif offset_x > x_adj:
                    offset_x = x_adj
                    outside = True

                if offset_y < 0:
                    offset_y = 0
                    outside = True
                elif offset_y > y_adj:
                    offset_y = y_adj
                    outside = True

                value_change = arr_change[offset_x, offset_y, offset_z]
                value_base = arr_base[offset_x, offset_y, offset_z]

                if outside or (
                    nodata
                    and (value_change == nodata_value or value_base == nodata_value)
                ):
                    continue

                hood_values_change[n] = value_change
                hood_values_base[n] = value_base

                weight = weights[n]

                hood_weights[n] = weight
                weight_sum[0] += weight

            hood_weights = np.divide(hood_weights, weight_sum[0])

            if weight_sum[0] > 0:
                hood_change_median = hood_quantile(
                    hood_values_change, hood_weights, quantile
                )
                hood_change_mad = hood_mad(hood_values_change, hood_weights)
                hood_base_median = hood_quantile(
                    hood_values_base, hood_weights, quantile
                )
                hood_base_mad = hood_mad(hood_values_base, hood_weights)

                result[x][y] = (
                    np.true_divide(
                        (value_change - hood_change_median) * hood_change_mad,
                        hood_base_mad,
                    )
                    + hood_base_median
                )

    return result


@jit(nopython=True, parallel=True, nogil=True, fastmath=True, cache=True)
def median_collapse(
    arr,
    offsets,
    weights,
    quantile=0.5,
    nodata=True,
    nodata_value=-9999.0,
    weighted=True,
):
    x_adj = arr.shape[0] - 1
    y_adj = arr.shape[1] - 1
    z_adj = (arr.shape[2] - 1) // 2

    hood_size = len(offsets)
    if nodata:
        result = np.full(arr.shape[:2], nodata_value, dtype="float32")
    else:
        result = np.zeros(arr.shape[:2], dtype="float32")

    for x in prange(arr.shape[0]):
        for y in range(arr.shape[1]):

            hood_values = np.zeros(hood_size, dtype="float32")
            hood_weights = np.zeros(hood_size, dtype="float32")
            weight_sum = np.array([0.0], dtype="float32")

            for n in range(hood_size):
                offset_x = x + offsets[n][0]
                offset_y = y + offsets[n][1]
                offset_z = offsets[n][2]

                outside = False

                if offset_z < -z_adj:
                    offset_z = -z_adj
                    outside = True
                elif offset_z > z_adj:
                    offset_z = z_adj
                    outside = True

                if offset_x < 0:
                    offset_x = 0
                    outside = True
                elif offset_x > x_adj:
                    offset_x = x_adj
                    outside = True

                if offset_y < 0:
                    offset_y = 0
                    outside = True
                elif offset_y > y_adj:
                    offset_y = y_adj
                    outside = True

                value = arr[offset_x, offset_y, offset_z]

                if outside or (nodata and value == nodata_value):
                    continue

                hood_values[n] = value
                weight = weights[n]

                hood_weights[n] = weight
                weight_sum[0] += weight

            hood_weights = np.divide(hood_weights, weight_sum[0])

            if weight_sum[0] > 0:
                if weighted:
                    result[x, y] = hood_quantile(hood_values, hood_weights, quantile)
                else:
                    result[x, y] = np.median(hood_values[np.nonzero(hood_weights)])

    return result


folder = "C:/Users/caspe/Desktop/test_area2/"
layers = [
    folder + "vv_01.tif",
    folder + "vv_02.tif",
    folder + "vv_03.tif",
]

sar_data = raster_to_array(layers)

kernel_size = 3
nodata_value = -9999.0

_kernel, offsets_3D, weights_3D = create_kernel(
    (kernel_size, kernel_size, sar_data.shape[2]),
    distance_calc=False,  # "gaussian"
    sigma=1,
    spherical=True,
    radius_method="ellipsoid",
    offsets=True,
    edge_weights=True,
    normalised=True,
    remove_zero_weights=True,
)

_kernel, offsets_2D, weights_2D = create_kernel(
    (kernel_size, kernel_size, 1),
    distance_calc=False,  # "gaussian"
    spherical=True,
    radius_method="ellipsoid",
    offsets=True,
    edge_weights=True,
    normalised=True,
    remove_zero_weights=True,
)

# panned1 = mad_match(
#     raster_to_array(layers[0]),
#     raster_to_array(layers[1]),
#     offsets_2D,
#     weights_2D,
# )
# panned3 = mad_match(
#     raster_to_array(layers[2]),
#     raster_to_array(layers[1]),
#     offsets_2D,
#     weights_2D,
# )

# array_to_raster(panned1, folder + "vv_01.tif", out_path=folder + "01_panned.tif")
# array_to_raster(panned3, folder + "vv_03.tif", out_path=folder + "03_panned.tif")

sar_data = raster_to_array(
    [
        folder + "01_panned.tif",
        folder + "vv_02.tif",
        folder + "03_panned.tif",
    ]
)

result = median_collapse(
    sar_data,
    offsets_3D,
    weights_3D,
    weighted=True,
    nodata_value=nodata_value,
    nodata=True,
)

array_to_raster(result, folder + "vv_01.tif", out_path=folder + "elips_median_mad.tif")
