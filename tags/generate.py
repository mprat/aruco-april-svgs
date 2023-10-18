# dir(cv2.aruco) --> everything that starts with DICT_

import os

import cv2
import drawsvg as draw
import numpy as np


def bit_coordinate_to_corner_coordinate(bit_coordinate):
    return [
        tuple(bit_coordinate),
        tuple(bit_coordinate + np.array([0, 1])),
        tuple(bit_coordinate + np.array([1, 1])),
        tuple(bit_coordinate + np.array([1, 0])),
    ]


def generate_svg(
    tag_dict,
    tag_id,
    border_bits=1,
    save_folder="output",
    basename="test",
    save=True,
    step=100,
):
    bits_list = cv2.aruco.Dictionary.getBitsFromByteList(
        tag_dict.bytesList[tag_id].flatten()[
            0 : int(np.ceil(tag_dict.markerSize * tag_dict.markerSize / 8))
        ],
        tag_dict.markerSize,
    )
    grid_size = tag_dict.markerSize + 2 * border_bits

    # from each bit list, find each connected component
    # of positive bits. draw a line group connecting each of
    # these so there are the minimal number of SVG components
    # per group
    tag_drawing = draw.Drawing(
        step * grid_size,
        step * grid_size,
        origin=(0, 0),
        displayInline=False,
    )

    tag_group = draw.Group()
    tag_group.append(
        draw.Rectangle(
            x=0, y=0, width=step * grid_size, height=step * grid_size, fill="black"
        )
    )

    num_components, components = cv2.connectedComponents(bits_list, connectivity=4)
    line_corners = []

    # component id 0 is where there are no tags present
    for component_id in range(1, num_components):
        corners = []
        # go through and add all the coordinates and add the
        # corners of the connected component. the coordinates
        # are defined like this (e.g. for a 4x4 tag)
        #
        #   0   1   2   3   4
        # 0 +---+---+---+---+
        #   |   |   |   |   |
        # 1 +---+---+---+---+
        #   |   |   |   |   |
        # 2 +---+---+---+---+
        #   |   |   |   |   |
        # 3 +---+---+---+---+
        #   |   |   |   |   |
        # 4 +---+---+---+---+
        #
        # where the tag squares are defined like this
        #     0   1   2   3
        #   +---+---+---+---+
        # 0 |   |   |   |   |
        #   +---+---+---+---+
        # 1 |   |   |   |   |
        #   +---+---+---+---+
        # 2 |   |   |   |   |
        #   +---+---+---+---+
        # 3 |   |   |   |   |
        #   +---+---+---+---+
        #
        tag_square_coords = np.array(list(zip(*np.where(components == component_id))))
        candidates = set(bit_coordinate_to_corner_coordinate(tag_square_coords[0]))
        for sq in tag_square_coords[1:]:
            candidates = candidates ^ set(bit_coordinate_to_corner_coordinate(sq))

        if len(candidates) > 4:
            import pdb

            pdb.set_trace()

        first = candidates.pop()
        can_list = list(candidates)
        ordered = [first]
        while len(can_list) > 0:
            for can in can_list:
                if can[0] == first[0] or can[1] == first[1]:
                    break
            can_list.remove(can)
            ordered.append(can)
            first = can

        assert ordered[0][0] == ordered[-1][0] or ordered[0][1] == ordered[-1][1]

        # to each of the coordinates we need to add the border_bits
        line_corners.append(np.fliplr(np.array(ordered)) + border_bits)

    for corners in line_corners:
        tag_group.append(
            draw.Lines(
                *(np.array(corners)).flatten() * step,
                close=True,
                stroke_width=2,
                fill="white",
                fill_opacity=1,
                stroke="white",
            )
        )

    tag_drawing.append(tag_group)
    if save:
        # make sure the folder exists
        os.makedirs(save_folder, exist_ok=True)
        tag_drawing.set_render_size(w="1.75in", h="1.75in")
        tag_drawing.save_svg(os.path.join(save_folder, f"{basename}_{tag_id:04}.svg"))
    return tag_drawing


if __name__ == "__main__":
    dict_name = cv2.aruco.DICT_5X5_1000
    tag_dict = cv2.aruco.getPredefinedDictionary(dict_name)
    basename = "DICT_5X5_1000"

    # for tag_id in range(0, tag_dict.bytesList.shape[0]):
    for tag_id in range(1, 2):
        try:
            generate_svg(
                tag_dict=tag_dict,
                tag_id=tag_id,
                border_bits=1,
                save_folder="output",
                basename=basename,
                save=True,
            )
        except:
            print(f"failed on tag {tag_id}")
