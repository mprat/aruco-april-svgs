"""Helper functions related to tag dictionaries."""

from typing import List, NamedTuple

import cv2


class TagDict(NamedTuple):
    """Encapsulate an OpenCV aruco dictionary with a name."""

    dictionary: cv2.aruco.Dictionary
    name: str


def get_dict(dict_name: str) -> TagDict:
    """Turn a string into an OpenCV aruco dictionary."""
    if dict_name == "DICT_4X4_50":
        dict_name = cv2.aruco.DICT_4X4_50
    elif dict_name == "DICT_4X4_100":
        dict_name = cv2.aruco.DICT_4X4_100
    elif dict_name == "DICT_4X4_250":
        dict_name = cv2.aruco.DICT_4X4_250
    elif dict_name == "DICT_4X4_1000":
        dict_name = cv2.aruco.DICT_4X4_1000
    elif dict_name == "DICT_5X5_50":
        dict_name = cv2.aruco.DICT_5X5_50
    elif dict_name == "DICT_5X5_100":
        dict_name = cv2.aruco.DICT_5X5_100
    elif dict_name == "DICT_5X5_250":
        dict_name = cv2.aruco.DICT_5X5_250
    elif dict_name == "DICT_5X5_1000":
        dict_name = cv2.aruco.DICT_5X5_1000
    elif dict_name == "DICT_6X6_50":
        dict_name = cv2.aruco.DICT_6X6_50
    elif dict_name == "DICT_6X6_100":
        dict_name = cv2.aruco.DICT_6X6_100
    elif dict_name == "DICT_6X6_250":
        dict_name = cv2.aruco.DICT_6X6_250
    elif dict_name == "DICT_6X6_1000":
        dict_name = cv2.aruco.DICT_6X6_1000
    elif dict_name == "DICT_7X7_50":
        dict_name = cv2.aruco.DICT_7X7_50
    elif dict_name == "DICT_7X7_100":
        dict_name = cv2.aruco.DICT_7X7_100
    elif dict_name == "DICT_7X7_250":
        dict_name = cv2.aruco.DICT_7X7_250
    elif dict_name == "DICT_7X7_1000":
        dict_name = cv2.aruco.DICT_7X7_1000
    elif dict_name == "DICT_APRILTAG_16H5":
        dict_name = cv2.aruco.DICT_APRILTAG_16H5
    elif dict_name == "DICT_APRILTAG_16h5":
        dict_name = cv2.aruco.DICT_APRILTAG_16h5
    elif dict_name == "DICT_APRILTAG_25H9":
        dict_name = cv2.aruco.DICT_APRILTAG_25H9
    elif dict_name == "DICT_APRILTAG_25h9":
        dict_name = cv2.aruco.DICT_APRILTAG_25h9
    elif dict_name == "DICT_APRILTAG_36H10":
        dict_name = cv2.aruco.DICT_APRILTAG_36H10
    elif dict_name == "DICT_APRILTAG_36H11":
        dict_name = cv2.aruco.DICT_APRILTAG_36H11
    elif dict_name == "DICT_APRILTAG_36h10":
        dict_name = cv2.aruco.DICT_APRILTAG_36h10
    elif dict_name == "DICT_APRILTAG_36h11":
        dict_name = cv2.aruco.DICT_APRILTAG_36h11
    elif dict_name == "DICT_ARUCO_MIP_36H12":
        dict_name = cv2.aruco.DICT_ARUCO_MIP_36H12
    elif dict_name == "DICT_ARUCO_MIP_36h12":
        dict_name = cv2.aruco.DICT_ARUCO_MIP_36h12
    elif dict_name == "DICT_ARUCO_ORIGINAL":
        dict_name = cv2.aruco.DICT_ARUCO_ORIGINAL
    else:
        raise RuntimeError(f"Unknown dict name {dict_name}")
    return TagDict(
        dictionary=cv2.aruco.getPredefinedDictionary(dict_name), name=dict_name
    )


def get_all_dict_names_opencv() -> List[str]:
    """Return all the names of dictionaries in OpenCV."""
    return [dict_name for dict_name in dir(cv2.aruco) if "DICT_" in dict_name]
