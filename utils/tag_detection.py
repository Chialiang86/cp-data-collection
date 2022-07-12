import cv2
import numpy as np
from dt_apriltags import Detector


at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)


def detect_tags(img, cam_at, tag_size, tag_list=[], visualize=True):
    origin_img = img.copy()
    draw_img = origin_img.copy()

    def draw_tag(tag):
        # tag sides
        for idx in range(len(tag.corners)):
            cv2.line(draw_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0))

        # tag ID
        cv2.putText(draw_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))

    # apriltag detector
    gray_img = cv2.cvtColor(origin_img, cv2.COLOR_RGB2GRAY)
    tags = at_detector.detect(gray_img, False, cam_at, tag_size)

    tag_IDs, tag_img_pts = [], []
    for tag in tags:
        if len(tag_list) == 0 or tag.tag_id in tag_list:
            # find tag ID
            tag_IDs.append(tag.tag_id)

            # find tag img pts
            img_pts = np.array([pt for pt in tag.corners])
            img_pts = np.insert(img_pts, 0, tag.center, axis=0)
            tag_img_pts.append(img_pts)

            if visualize:
                draw_tag(tag)

    return draw_img, tag_IDs, tag_img_pts


def get_tagboard_obj_pts(tagboard_dict, tag_IDs):
    # return np.array(list(tagboard_dict.values()))[tag_IDs].reshape(-1, 3)
    tagboard_obj_pts = []
    for id in tag_IDs:
        tagboard_obj_pts.append(tagboard_dict[id])

    return np.array(tagboard_obj_pts).reshape(-1, 3)


# board8x12
def four_tag_mappings(ld_ids, nx=8, ny=12, tsize=0.02, tspace=0.2, return_centers=False):
    """
    :param ld_ids: Left down indices for markers with four tags (20 points)
    :param nx: params for get_tag_board
    :param ny: params for get_tag_board
    :param tsize: params for get_tag_board
    :param space: params for get_tag_board
    :param return_centers: return the center position of each 4-tag marker (for evaluation purpose)
    :return: dictionary of location_mappings (key=ld_id)
    """
    board_map, _ = get_tag_board(nx, ny, tsize, tspace)
    location_mappings = {}
    centers = {}
    for id in ld_ids:  # for each four-tag marker
        mapping = {}
        tag_indices = [id, id + 1, id + nx, id + nx + 1]
        center = np.zeros(3)
        for t in tag_indices:
            mapping[t] = board_map[t]  # 5*3
            center += board_map[t][0]  # center point
        center /= 4.0
        for k in mapping.keys():
            mapping[k] = mapping[k] - center
        centers[id] = center
        location_mappings[id] = mapping
    if return_centers:
        return location_mappings, centers
    else:
        return location_mappings


def get_five_points(center, half):
    center_x, center_y = center
    five_points = np.zeros((5, 3))
    five_points[0] = [center_x, center_y, 0]
    five_points[1] = [center_x - half, center_y - half, 0]
    five_points[2] = [center_x + half, center_y - half, 0]
    five_points[3] = [center_x + half, center_y + half, 0]
    five_points[4] = [center_x - half, center_y + half, 0]
    return five_points


def get_tag_board(nx, ny, tsize, tspace):
    location_mapping = {}
    """
    id: 5*3 array, center,left_down,.(CCW)..,left_up corners 
    """
    id_arrangement = np.arange(nx * ny).reshape(ny, nx).tolist()
    half = tsize / 2.0
    spacing = tsize * tspace

    for row, r_tags in enumerate(id_arrangement):
        for col, c_tag in enumerate(r_tags):
            center_x = (spacing + tsize) * col + spacing + half
            center_y = (spacing + tsize) * row + spacing + half
            location_mapping[c_tag] = get_five_points((center_x, center_y), half)

    return location_mapping, tsize


def tag_boards(b_name):  # --show-id: show ids under the tags, --blue: tag in color blue
    if b_name == 'board10x14':  # python create_tag_pdf.py --type apriltag --nx 10 --ny 14 --tsize 0.018 --tspace 0.1
        return get_tag_board(10, 14, 0.018, 0.1)
    elif b_name == 'board9x13':  # python create_tag_pdf.py --type apriltag --nx 9 --ny 13 --tsize 0.02 --tspace 0.125
        return get_tag_board(nx=9, ny=13, tsize=0.02, tspace=0.125)
    elif b_name == 'board8x12_small':  # python create_tag_pdf.py --type apriltag --nx 8 --ny 12 --tsize 0.018 --tspace 0.2
        return get_tag_board(nx=8, ny=12, tsize=0.018, tspace=0.2)
    elif b_name == 'board8x12':  # python create_tag_pdf.py --type apriltag --nx 8 --ny 12 --tsize 0.02 --tspace 0.2
        return get_tag_board(nx=8, ny=12, tsize=0.02, tspace=0.2)
    elif b_name == 'board6x9':  # python create_tag_pdf.py --type apriltag --nx 6 --ny 9 --tsize 0.025 --tspace 0.25
        return get_tag_board(nx=6, ny=9, tsize=0.025, tspace=0.25)
    elif b_name == 'board4x6':  # python create_tag_pdf.py  --type apriltag --nx 4 --ny 6 --tsize 0.04 --tspace 0.2
        return get_tag_board(nx=4, ny=6, tsize=0.04, tspace=0.2)
    elif b_name == 'board9x9_large':  # python create_tag_pdf.py --type apriltag --nx 9 --ny 9 --tsize 0.04 --tspace 0.25
        return get_tag_board(nx=9, ny=9, tsize=0.04, tspace=0.25)
    else:
        raise NotImplementedError