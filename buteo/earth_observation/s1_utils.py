import os
import xml.etree.ElementTree as ET
from datetime import datetime


def find_gpt(test_gpt_path):
    gpt = os.path.realpath(os.path.abspath(os.path.expanduser(test_gpt_path)))
    if not os.path.exists(gpt):
        gpt = os.path.realpath(
            os.path.abspath(os.path.expanduser("~/esa_snap/bin/gpt"))
        )
        if not os.path.exists(gpt):
            gpt = os.path.realpath(
                os.path.abspath(os.path.expanduser("~/snap/bin/gpt"))
            )
            if not os.path.exists(gpt):
                gpt = os.path.realpath(
                    os.path.abspath(os.path.expanduser("/opt/esa_snap/bin/gpt"))
                )
                if not os.path.exists(gpt):
                    gpt = os.path.realpath(
                        os.path.abspath(os.path.expanduser("/opt/snap/bin/gpt"))
                    )
                    if not os.path.exists(gpt):
                        gpt = os.path.realpath(
                            os.path.abspath(
                                os.path.expanduser("C:/Program Files/snap/bin/gpt.exe")
                            )
                        )

                        if os.path.exists(gpt):
                            gpt = '"C:/Program Files/snap/bin/gpt.exe"'
                        else:
                            if not os.path.exists(gpt):
                                assert os.path.exists(
                                    gpt
                                ), "Graph processing tool not found."

    return gpt


# TODO: wkt to ogr geometry
def s1_kml_to_bbox(path_to_kml):
    root = ET.parse(path_to_kml).getroot()
    for elem in root.iter():
        if elem.tag == "coordinates":
            coords = elem.text
            break

    coords = coords.split(",")
    coords[0] = coords[-1] + " " + coords[0]
    del coords[-1]
    coords.append(coords[0])

    min_x = 180
    max_x = -180
    min_y = 90
    max_y = -90

    for i in range(len(coords)):
        intermediate = coords[i].split(" ")
        intermediate.reverse()

        intermediate[0] = float(intermediate[0])
        intermediate[1] = float(intermediate[1])

        if intermediate[0] < min_x:
            min_x = intermediate[0]
        elif intermediate[0] > max_x:
            max_x = intermediate[0]

        if intermediate[1] < min_y:
            min_y = intermediate[1]
        elif intermediate[1] > max_y:
            max_y = intermediate[1]

    footprint = f"POLYGON (({min_x} {min_y}, {min_x} {max_y}, {max_x} {max_y}, {max_x} {min_y}, {min_x} {min_y}))"

    return footprint


# TODO: get_metadata_from_zip
def get_metadata(image_paths):
    images_obj = []

    for img in image_paths:
        kml = f"{img}/preview/map-overlay.kml"

        timestr = str(img.rsplit(".")[0].split("/")[-1].split("_")[5])
        timestamp = datetime.strptime(timestr, "%Y%m%dT%H%M%S").timestamp()

        meta = {
            "path": img,
            "timestamp": timestamp,
            "footprint_wkt": s1_kml_to_bbox(kml),
        }

        images_obj.append(meta)

    return images_obj
