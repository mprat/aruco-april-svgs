# dir(cv2.aruco) --> everything that starts with DICT_

import cv2
import drawsvg as draw
import numpy as np


dict_name = cv2.aruco.DICT_4X4_50
tag_id = 42
border_bits = 1
tag_dict = cv2.aruco.getPredefinedDictionary(dict_name)
bits_list = cv2.aruco.Dictionary.getBitsFromByteList(
    tag_dict.bytesList[tag_id].flatten()[0 : int(np.ceil(tag_dict.markerSize // 2))],
    tag_dict.markerSize,
)
grid_size = tag_dict.markerSize + 2 * border_bits

# from each bit list, find each connected component
# of positive bits. draw a line group connecting each of
# these so there are the minimal number of SVG components
# per group
step = 100
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


def bit_coordinate_to_corner_coordinate(bit_coordinate):
    return [
        tuple(bit_coordinate),
        tuple(bit_coordinate + np.array([0, 1])),
        tuple(bit_coordinate + np.array([1, 1])),
        tuple(bit_coordinate + np.array([1, 0])),
    ]


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
    line_corners.append(np.fliplr(np.array(ordered)) + 1)

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
tag_drawing.save_svg("test.svg")
