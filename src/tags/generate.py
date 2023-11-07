# dir(cv2.aruco) --> everything that starts with DICT_

from enum import IntEnum
import os
import traceback
from typing import List, NamedTuple, Tuple

import cv2
import drawsvg as draw
import numpy as np

from .dictionary import TagDict, get_dict


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


class PathOrientation(IntEnum):
    INNER = 0
    OUTER = 1


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


def flip_orientation(seg):
    """Flip the orientation of the segment."""
    if seg.inside == Orientation.LEFT:
        seg = seg._replace(inside=Orientation.RIGHT)
    elif seg.inside == Orientation.RIGHT:
        seg = seg._replace(inside=Orientation.LEFT)
    elif seg.inside == Orientation.UP:
        seg = seg._replace(inside=Orientation.DOWN)
    elif seg.inside == Orientation.DOWN:
        seg = seg._replace(inside=Orientation.UP)
    return seg


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


def get_next_candidate_segments(
    current_segment: Segment,
    possible_segments: List[Segment],
):
    return [
        cand
        for cand in possible_segments
        if current_segment.start in (cand.start, cand.end)
        and not is_opposite(current_segment.inside, cand.inside)
    ]


def get_most_likely_next_segment(
    current_segment: Segment,
    next_candidates: List[Segment],
):
    # if we have only one option, that's our option!
    if len(next_candidates) == 1:
        return next_candidates[0]

    # otherwise we pick our most likely candidate and save the state
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
    cur_inside_point = get_inside_point(current_segment)
    inside_points = [get_inside_point(cand) for cand in next_candidates]
    inside_point_dist = [
        np.sqrt((ip[0] - cur_inside_point[0]) ** 2 + (ip[1] - cur_inside_point[1]) ** 2)
        for ip in inside_points
    ]
    return next_candidates[np.argmin(inside_point_dist)]


def get_next_path(
    segments: List[Segment], next_segment: Segment, path_orientation: PathOrientation
):
    """Return the next path from the list of segments.

    The caller of this function decides whether the segment
    is a hole, an outer, etc.

    For segments that have the desired path on the "outside", we
    flip the orientations of all the segments so that the logic
    for getting the next best segment is correct, then flip all
    the orientations back before returning the next segments to
    keep.
    """
    next_seg = next_segment  # TODO: .copy()
    path = [next_seg]
    remaining_segments = segments.copy()
    save_state = None

    while len(remaining_segments) > 0:
        next_seg_cands = get_next_candidate_segments(
            current_segment=next_seg, possible_segments=remaining_segments
        )

        if len(next_seg_cands) == 0:
            # if there are zero segments available then we have reached
            # a state where we need to backtrack our search or there
            # is a hole in the center. either way, this function needs
            # to return
            return path, remaining_segments, save_state

        # at this point, if we are looking for a PathOrientation of
        # OUTER, we need to flip t he orientations of all the candidates
        # this means we're likely looking for holes, and need to apply all
        # our heuristics "backwards"
        if path_orientation == PathOrientation.OUTER:
            next_seg = flip_orientation(next_seg)
            next_seg_cands = [flip_orientation(seg) for seg in next_seg_cands]

        next_seg = get_most_likely_next_segment(
            current_segment=next_seg, next_candidates=next_seg_cands
        )

        if path_orientation == PathOrientation.OUTER:
            next_seg = flip_orientation(next_seg)
            next_seg_cands = [flip_orientation(seg) for seg in next_seg_cands]

        # save the state
        # (segments not yet assigned,
        #  segments already assigned,
        #  segment about to try assigning,
        #  possible segments that can be assigned)
        # the one we've tried assigning is not yet removed from keep_segments
        next_seg_cands.remove(next_seg)
        save_state = (
            remaining_segments.copy(),
            path.copy(),
            [next_seg],
            next_seg_cands,
        )

        path.append(next_seg)
        remaining_segments.remove(next_seg)

    # if we get here, we're expecting the remaining_segments
    # variable to be empty, since we've used all the available
    # segments to make a path
    return path, remaining_segments, save_state


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
    all_holes = []

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
            np.fliplr(
                np.array(list(zip(*np.where(components == component_id), strict=True)))
            )
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
        #
        holes = []
        next_seg = keep_segments.pop()
        ordered, remaining_segments, save_state = get_next_path(
            segments=keep_segments,
            next_segment=next_seg,
            path_orientation=PathOrientation.INNER,
        )
        assert ordered[0].start in (ordered[-1].start, ordered[-1].end) or ordered[
            0
        ].end in (
            ordered[-1].start,
            ordered[-1].end,
        )
        line_corners.append(ordered_segments_to_corners_list(ordered))

        while len(remaining_segments) != 0:
            # we either need to make a hole, or move backwards
            # in the DFS search
            # have we reached the end of the path?
            if share_corner(ordered[0], ordered[-1]):
                # we believe the remaining path elements to be
                # a hole (or multiple holes!), so we need to reconnect that path
                next_seg = remaining_segments.pop()
                ordered, remaining_segments, save_state = get_next_path(
                    segments=remaining_segments,
                    next_segment=next_seg,
                    path_orientation=PathOrientation.OUTER,
                )

                # if this is FALSE and we have a save_state, then we need
                # to back-track and try a different candidate
                hole_valid = ordered[0].start in (
                    ordered[-1].start,
                    ordered[-1].end,
                ) or ordered[0].end in (
                    ordered[-1].start,
                    ordered[-1].end,
                )

                if hole_valid:
                    holes.append(ordered)
                else:
                    raise RuntimeError("Hole is not valid")
            else:
                raise RuntimeError("Not the end of a path, need to backtrack")

        if holes:
            hole_corners = []
            for hole in holes:
                hole_corners.append(ordered_segments_to_corners_list(hole))
            all_holes.append(hole_corners)
        else:
            all_holes.append(None)

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

    for corners, holes in zip(line_corners, all_holes, strict=True):
        if holes is not None:
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

            for hole in holes:
                # TODO: make sure this is in the opposite orientation from the line
                # it looks like this is already done?
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


def generate_all(tag_dict: TagDict, save_folder: str = "output", border_bits: int = 1):
    failed_tag_ids = []
    total = 0
    for tag_id in range(0, tag_dict.dictionary.bytesList.shape[0]):
        print(f"working on tag {tag_id}")
        total += 1
        try:
            generate_svg(
                tag_dict=tag_dict.dictionary,
                tag_id=tag_id,
                border_bits=border_bits,
                save_folder=save_folder,
                basename=tag_dict.name,
                save=True,
            )
        except Exception:
            print(f"failed on tag {tag_id}")
            print(traceback.format_exc())
            failed_tag_ids.append(tag_id)

    return tag_dict, total, failed_tag_ids


if __name__ == "__main__":
    tag_dict = get_dict(dict_name="DICT_5X5_1000")
    _, total, failed_tag_ids = generate_all(tag_dict=tag_dict.dictionary)

    print(f"Failed on {len(failed_tag_ids)} out of {total} tags")
    print(f"Failed: {failed_tag_ids}")
