import glob
import os

import cairosvg
import cv2

from tags import generate


def test_output_tags():
    """Generate all tags, convert to png, make sure we can detect."""
    # TODO: save to temp folder
    save_folder = "output"
    tag_dict, total, failed_tag_ids = generate.generate_all(
        basename="DICT_5X5_1000", save_folder=save_folder, border_bits=1
    )

    # now read all the tags back, convert to PNG, and make sure
    # we can detect them with the opencv aruco detector
    for svgpath in sorted(glob.glob(os.path.join(save_folder, "*.svg"))):
        pngpath = svgpath.replace(".svg", ".png")
        cairosvg.svg2png(url=svgpath, write_to=pngpath)

    # load and detect
    for pngpath in sorted(glob.glob(os.path.join(save_folder, "*.png"))):
        expected_tag_id = int(pngpath.split("_")[-1].replace(".png", ""))
        print(f"working on tag {expected_tag_id}")
        img = cv2.imread(pngpath)
        corners, ids, _ = cv2.aruco.detectMarkers(img, tag_dict)
        assert expected_tag_id == ids, expected_tag_id
