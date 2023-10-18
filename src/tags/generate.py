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
        (or1 == Orientation.LEFT and or2 == Orientation.RIGHT)
        or (or1 == Orientation.RIGHT and or2 == Orientation.LEFT)
        or (or1 == Orientation.UP and or2 == Orientation.DOWN)
        or (or1 == Orientation.DOWN and or2 == Orientation.UP)
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


def get_inside_point(seg):
    """The inside point is the point representing the "inside".

    It is at the midpoint of the segment, then 0.5 units away
    from the segment in the direction of the inside.
    """
    midpoint = ((seg.start[0] + seg.end[0]) / 2, (seg.start[1] + seg.end[1]) / 2)
    if seg.inside == Orientation.UP:
        return (midpoint[0], midpoint[1] - 0.5)
    if seg.inside == Orientation.DOWN:
        return (midpoint[0], midpoint[1] + 0.5)
    if seg.inside == Orientation.LEFT:
        return (midpoint[0] - 0.5, midpoint[1])
    if seg.inside == Orientation.RIGHT:
        return (midpoint[0] + 0.5, midpoint[1])


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
                and not is_opposite(next_seg.inside, cand.inside)
            ]

            # if there are zero segments available then we have reached
            # the DFS search or we have a hole in the center
            if len(next_seg_cands) == 0:
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
                            and not is_opposite(next_seg.inside, cand.inside)
                        ]
                        if len(next_seg_cands) != 1:
                            raise RuntimeError("WHEN")

                        next_seg = next_seg_cands[0]
                        hole.append(next_seg)
                        keep_segments.remove(next_seg)

                    # if this is FALSE and we have a save_state, then we need
                    # to back-track and try a different candidate
                    hole_valid = hole[0].start in (
                        hole[-1].start,
                        hole[-1].end,
                    ) or hole[0].end in (
                        hole[-1].start,
                        hole[-1].end,
                    )
                    if not hole_valid:
                        if save_state is not None:
                            # BACKTRACK from save_state
                            (
                                keep_segments,
                                ordered,
                                tried,
                                others_to_try,
                            ) = save_state

                            next_seg = others_to_try[0]
                            tried.append(next_seg)
                            others_to_try.remove(next_seg)

                            save_state = (
                                keep_segments,
                                ordered,
                                tried,
                                others_to_try,
                            )

                            import pdb

                            pdb.set_trace()
                            ordered.append(next_seg)
                            keep_segments.remove(next_seg)

                            # reset the hole
                            hole = []
                        else:
                            raise RuntimeError("WHERE")

                    # go back to the beginning of the loop
                    # since the hole is valid and we found one
                    continue

            if len(next_seg_cands) == 1:
                next_seg = next_seg_cands[0]
            else:
                # we pick our most likely candidate and save the state
                # our most likely candidate is the one that is "closer"
                # to the segment in question based on the "inside-ness" of
                # the segment. e.g.:
                #
                #   | ->   --- one we're matching to
                #
                #   ^
                #   |      --- candidate 1
                #   _
                #
                #   _
                #   |      --- candidate 2
                #  \ /
                #
                #  candidate 1 is "closer" because the point representing the orientation
                #  is closer to the one we're matching than the one from candidate 2
                #
                cur_inside_point = get_inside_point(next_seg)
                inside_points = [get_inside_point(cand) for cand in next_seg_cands]
                inside_point_dist = [
                    np.sqrt(
                        (ip[0] - cur_inside_point[0]) ** 2
                        + (ip[1] - cur_inside_point[1]) ** 2
                    )
                    for ip in inside_points
                ]
                next_seg = next_seg_cands[np.argmin(inside_point_dist)]

                # save the state
                # (segments not yet assigned,
                #  segments already assigned,
                #  segment about to try assigning,
                #  possible segments that can be assigned)
                # the one we've tried assigning is not yet removed from keep_segments
                next_seg_cands.remove(next_seg)
                save_state = (
                    keep_segments.copy(),
                    ordered.copy(),
                    [next_seg],
                    next_seg_cands,
                )

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

    # TODO: instead of 10, make sure it's at least 1, but also 1/10 of the step
    # we need to make sure that the tag itself has a tiny border around it so the
    # detection works, because it's not a guarantee that anything the tag is placed
    # on will be of a different color
    buffer = 10
    tag_orig = buffer // 2
    tag_drawing = draw.Drawing(
        step * grid_size + buffer,
        step * grid_size + buffer,
        origin=(0, 0),
        displayInline=False,
    )

    tag_group = draw.Group()
    tag_group.append(
        draw.Rectangle(
            x=0,
            y=0,
            width=step * grid_size + buffer,
            height=step * grid_size + buffer,
            fill="white",
        )
    )
    tag_group.append(
        draw.Rectangle(
            x=tag_orig,
            y=tag_orig,
            width=step * grid_size,
            height=step * grid_size,
            fill="black",
        )
    )

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
            path.M(corners[0][0] * step + tag_orig, corners[0][1] * step + tag_orig)
            for i in range(len(corners)):
                path.L(corners[i][0] * step + tag_orig, corners[i][1] * step + tag_orig)
            path.Z()

            # draw the hole
            path.M(hole[0][0] * step + tag_orig, hole[0][1] * step + tag_orig)
            for i in range(len(hole)):
                path.L(hole[i][0] * step + tag_orig, hole[i][1] * step + tag_orig)
            path.Z()

            tag_group.append(path)
        else:
            tag_group.append(
                draw.Lines(
                    *(np.array(corners)).flatten() * step + tag_orig,
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


def generate_all(basename, save_folder="output", border_bits=1):
    if basename == "DICT_5X5_1000":
        dict_name = cv2.aruco.DICT_5X5_1000
    else:
        raise RuntimeError(f"Unknown basename {basename}")
    tag_dict = cv2.aruco.getPredefinedDictionary(dict_name)
    failed_tag_ids = []
    total = 0

    for tag_id in range(0, tag_dict.bytesList.shape[0]):
        print(f"working on tag {tag_id}")
        total += 1
        try:
            generate_svg(
                tag_dict=tag_dict,
                tag_id=tag_id,
                border_bits=border_bits,
                save_folder=save_folder,
                basename=basename,
                save=True,
            )
        except Exception as exc:
            print(f"failed on tag {tag_id}")
            print(traceback.format_exc())
            failed_tag_ids.append(tag_id)

    return tag_dict, total, failed_tag_ids


if __name__ == "__main__":
    _, total, failed_tag_ids = generate_all(basename="DICT_5X5_1000")

    print(f"Failed on {len(failed_tag_ids)} out of {total} tags")
    print(f"Failed: {failed_tag_ids}")
