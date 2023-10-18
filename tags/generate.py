# dir(cv2.aruco) --> everything that starts with DICT_

from enum import IntEnum
import os
import traceback
from typing import NamedTuple, Tuple

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


class Orientation(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


def is_opposite(or1, or2):
    return (
        (or1.LEFT and or2.RIGHT)
        or (or1.RIGHT and or2.LEFT)
        or (or1.UP and or2.DOWN)
        or (or1.DOWN and or2.UP)
    )


class Segment(NamedTuple):
    start: Tuple
    end: Tuple
    inside: Orientation


def is_same(seg1, seg2):
    return (seg1.start == seg2.start and seg1.end == seg2.end) or (
        seg1.start == seg2.end and seg1.end == seg2.start
    )


def share_corner(seg1, seg2):
    return (
        seg1.start == seg2.start
        or seg1.start == seg2.end
        or seg1.end == seg2.start
        or seg1.end == seg2.end
    )


def tag_square_to_segments(tag_coord):
    segments = []
    segments.append(
        Segment(
            start=(tag_coord[0], tag_coord[1]),
            end=(tag_coord[0] + 1, tag_coord[1]),
            inside=Orientation.DOWN,
        )
    )
    segments.append(
        Segment(
            start=(tag_coord[0] + 1, tag_coord[1]),
            end=(tag_coord[0] + 1, tag_coord[1] + 1),
            inside=Orientation.LEFT,
        )
    )
    segments.append(
        Segment(
            start=(tag_coord[0] + 1, tag_coord[1] + 1),
            end=(tag_coord[0], tag_coord[1] + 1),
            inside=Orientation.UP,
        )
    )
    segments.append(
        Segment(
            start=(tag_coord[0], tag_coord[1] + 1),
            end=(tag_coord[0], tag_coord[1]),
            inside=Orientation.RIGHT,
        )
    )
    return segments


def ordered_segments_to_corners_list(segments):
    corners_list = []

    # the first corner is the one that is shared between the
    # start and end segments list
    if segments[0].start in (segments[-1].start, segments[-1].end):
        corners_list.append(segments[0].end)
    else:
        corners_list.append(segments[0].start)

    for segment in segments[1:]:
        if corners_list[-1] == segment.start:
            corners_list.append(segment.end)
        else:
            corners_list.append(segment.start)

    return corners_list


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
    holes = []

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
        # add the border bits from the get-go so we don't forget!
        # we also fliplr here so that X and Y are correctly drawn
        # in the SVG coordinate system
        tag_square_coords = (
            np.fliplr(np.array(list(zip(*np.where(components == component_id)))))
            + border_bits
        )
        tag_square_coords = [tuple(tsc) for tsc in tag_square_coords]

        segments = []
        for tag_coord in tag_square_coords:
            segments.extend(tag_square_to_segments(tag_coord))
        keep_segments = []
        while len(segments) > 0:
            current = segments.pop()
            removed = False

            same_segments = [seg for seg in segments if is_same(seg, current)]
            for same_cand in same_segments:
                if is_opposite(current.inside, same_cand.inside):
                    # we need to remove both segments
                    # namely, we have not yet added current
                    # to any kept segments, so we don't add it
                    # (and it is already removed from segments)
                    # and then we need to remove the same_cand
                    # from segments as well
                    segments.remove(same_cand)
                    removed = True

            # if we didn't remove the segment (i.e. we don't have an
            # equivalent segment) then this segment is unique and we
            # can keep it. if we did remove a segment, then we'll keep going
            # with the loop until there aren't any more candidate
            # segments to go through
            if not removed:
                keep_segments.append(current)

        # now we need to re-order the kept segments so that
        # it is a consistent path with the insides / outsides
        # being preserved properly. there are only 3 kinds of
        # segments that can come next based on the existing
        # segment rules
        next_seg = keep_segments.pop()
        ordered = [next_seg]
        hole = []
        while len(keep_segments) > 0:
            next_seg_cands = [
                cand
                for cand in keep_segments
                if next_seg.start in (cand.start, cand.end)
            ]

            if len(next_seg_cands) != 1:
                # this happens when there is a square that touches another square in a corner
                # but is also connected to the rest of the shape (e.g. DICT_5X5_1000 tag 0)
                # in this case we pick one, then if we have tags left over at the end that
                # we can't match, we need to
                if len(next_seg_cands) == 3:
                    next_seg = next_seg_cands[0]
                    save_state = (keep_segments.copy(), ordered.copy())
                    # TODO: this hasn't come up yet?
                    #
                    ordered.append(next_seg)
                    keep_segments.remove(next_seg)
                elif len(next_seg_cands) == 0:
                    # we have no candidates left, so we either
                    # have reached the end of the DFS from the previous
                    # state, or we have a hole in the center of this shape.

                    # have we reached the end of the path?
                    if share_corner(ordered[0], ordered[-1]):
                        # we believe the remaining path elements to be
                        # a hole, so we need to reconnect that path
                        next_seg = keep_segments.pop()
                        hole = [next_seg]
                        while len(keep_segments) > 0:
                            next_seg_cands = [
                                cand
                                for cand in keep_segments
                                if next_seg.start in (cand.start, cand.end)
                            ]

                            if len(next_seg_cands) != 1:
                                raise RuntimeError("WHEN")

                            next_seg = next_seg_cands[0]
                            hole.append(next_seg)
                            keep_segments.remove(next_seg)

                        assert hole[0].start in (
                            hole[-1].start,
                            hole[-1].end,
                        ) or hole[0].end in (
                            hole[-1].start,
                            hole[-1].end,
                        )
                    else:
                        # we've reached a failed DFS state and need to re-route backwards
                        raise RuntimeError("WHO")
                else:
                    raise RuntimeError("WHAT")
            else:
                next_seg = next_seg_cands[0]
                ordered.append(next_seg)
                keep_segments.remove(next_seg)

        assert ordered[0].start in (ordered[-1].start, ordered[-1].end) or ordered[
            0
        ].end in (
            ordered[-1].start,
            ordered[-1].end,
        )

        line_corners.append(ordered_segments_to_corners_list(ordered))
        if hole:
            holes.append(ordered_segments_to_corners_list(hole))
        else:
            holes.append(None)

    # TODO: doesn't handle case of shape inside another shape. wonder where this will fail

    for corners, hole in zip(line_corners, holes):
        if hole is not None:
            # if False:
            # TODO: MAKE SURE THIS IS IN THE OPPOSITE ORIENTATION FROM THE LINE
            path = draw.Path(
                stroke_width=2,
                fill="white",
                fill_opacity=1,
                stroke="white",
            )
            # draw the line
            path.M(corners[0][0] * step, corners[0][1] * step)
            for i in range(len(corners)):
                path.L(corners[i][0] * step, corners[i][1] * step)
            path.Z()

            # draw the hole
            path.M(hole[0][0] * step, hole[0][1] * step)
            for i in range(len(hole)):
                path.L(hole[i][0] * step, hole[i][1] * step)
            path.Z()

            tag_group.append(path)
        else:
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
    failed_tag_ids = []
    total = 0

    for tag_id in range(0, tag_dict.bytesList.shape[0]):
        # for tag_id in range(3, 4):
        print(f"working on tag {tag_id}")
        total += 1
        try:
            generate_svg(
                tag_dict=tag_dict,
                tag_id=tag_id,
                border_bits=1,
                save_folder="output",
                basename=basename,
                save=True,
            )
        except Exception as exc:
            print(f"failed on tag {tag_id}")
            print(traceback.format_exc())
            failed_tag_ids.append(tag_id)

    print(f"Failed on {len(failed_tag_ids)} out of {total} tags")
    print(f"Failed: {failed_tag_ids}")
