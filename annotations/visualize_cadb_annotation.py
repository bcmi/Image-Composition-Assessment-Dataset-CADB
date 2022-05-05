import json
import cv2
import os
import numpy as np
import shutil
import argparse
from tqdm import tqdm
import math

score_levels = ['[1,2)', '[2,3)', '[3,4)', '[4,5]']
element_categories = ["center", "rule_of_thirds", "golden_ratio", "horizontal", "vertical", "diagonal",
              "curved", "fill_the_frame", "radial", "vanishing_point", "symmetric", "triangle",
              "pattern", 'none']
scene_categories = ['animal', 'plant', 'human', 'static', 'architecture',
                    'landscape', 'cityscape', 'indoor', 'night', 'other']

def draw_auxiliary_line(image, com_type):
    im_h, im_w, _ = image.shape
    if com_type in ['rule_of_thirds','center']:
        x1, x2 = int(im_w / 3), int(im_w / 3 * 2)
        y1, y2 = int(im_h / 3), int(im_h / 3 * 2)
    else:
        x1, x2 = int(im_w * 1. / 2.618), int(im_w * 1.618 / 2.618)
        y1, y2 = int(im_h * 1. / 2.618), int(im_h * 1.618 / 2.618)
    color  = (255, 255, 255)
    line_width = 3
    cv2.line(image, (0, y1), (im_w, y1), color, line_width)
    cv2.line(image, (0, y2), (im_w, y2), color, line_width)
    cv2.line(image, (x1, 0), (x1, im_h), color, line_width)
    cv2.line(image, (x2, 0), (x2, im_h), color, line_width)
    return image

def draw_element_on_image(image, com_type, element):
    pd_color = (0, 255, 255)
    if len(element) > 0:
        if com_type in ['center', 'rule_of_thirds', 'golden_ratio']:
            image = draw_auxiliary_line(image.copy(), com_type)
            for rect in element:
                x1,y1,x2,y2 = map(int, rect)
                cv2.rectangle(image, (x1,y1), (x2,y2), pd_color, 5)
        elif com_type in ['horizontal', 'diagonal', 'vertical']:
            image = draw_line_elements(image.copy(), com_type, element)
        else:
            element = np.array(element).astype(np.int32).reshape((len(element), -1, 2))
            for i in range(element.shape[0]):
                for j in range(element[i].shape[0] - 1):
                    cv2.line(image, (element[i, j, 0], element[i, j, 1]), (element[i, j + 1, 0], element[i, j + 1, 1]),
                             pd_color, 5)

    text = '{}'.format(com_type)
    cv2.putText(image, text, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, pd_color, 3)
    return image

def compute_angle(lines):
    lines = np.array(lines).reshape((-1,4))
    reverse_lines = np.concatenate([lines[:,2:], lines[:,0:2]], axis=1)
    l2r_points = np.where(lines[:,0:1] <= lines[:,2:3], lines, reverse_lines)
    angle = np.rad2deg(np.arctan2(l2r_points[:,3] - l2r_points[:,1], l2r_points[:,2] - l2r_points[:,0]))
    return np.abs(angle)

def draw_line_elements(src, comp, element, vis_angle=False):
    im_h, im_w, _ = src.shape
    angle = compute_angle(element)
    element = np.array(element).astype(np.int32).reshape((-1, 4))
    color = (0,255,255)
    for i in range(element.shape[0]):
        cv2.line(src, (element[i, 0], element[i, 1]),
                 (element[i, 2], element[i, 3]), color, 5)
        if vis_angle:
            text = '{:.1f}'.format(angle[i])
            tl_point = (element[i,0], element[i,1])
            br_point = (element[i,2], element[i,3])
            if element[i,0] > element[i,2] or \
                    (element[i,0] == element[i,2] and element[i,1] > element[i,3]):
                tl_point = (element[i,2], element[i,3])
                br_point = (element[i,0], element[i,1])
            if tl_point[1] <= br_point[1]:
                pos_x = max(min(tl_point[0] + 20, im_w - 50), 0)
                pos_y = max(tl_point[1] - 10, 30)
            else:
                pos_x = max(min(tl_point[0] - 100, im_w - 50), 0)
                pos_y = max(min(tl_point[1] + 50, im_h - 50), 0)
            cv2.putText(src, text, (pos_x, pos_y), cv2.FONT_HERSHEY_COMPLEX, 1.2, color, 2)
    return src

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root',default='./CADB_Dataset/',
                        help='path to images (should have subfolder images/)')
    parser.add_argument('--save_folder', default='./CADB_Dataset/Visualization/',type=str)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    opt, _ = parser.parse_known_args()
    image_dir = os.path.join(opt.data_root, 'images')
    assert os.path.exists(image_dir), image_dir
    element_file = os.path.join(opt.data_root, 'composition_elements.json')
    assert os.path.exists(element_file), element_file
    element_anno = json.load(open(element_file, 'r'))

    score_file = os.path.join(opt.data_root, 'composition_scores.json')
    assert os.path.exists(score_file), score_file
    score_anno = json.load(open(score_file, 'r'))

    scene_file = os.path.join(opt.data_root, 'scene_categories.json')
    assert os.path.exists(scene_file), scene_file
    scene_anno = json.load(open(scene_file, 'r'))

    score_dir = os.path.join(opt.save_folder, 'composition_scores')
    os.makedirs(score_dir, exist_ok=True)

    element_dir = os.path.join(opt.save_folder, 'composition_elements')
    os.makedirs(element_dir, exist_ok=True)

    scene_dir = os.path.join(opt.save_folder, 'scene_classification')
    os.makedirs(scene_dir, exist_ok=True)

    score_stats = {}
    for level in score_levels:
        score_stats[level] = []

    element_stats = {}
    for comp in element_categories:
        element_stats[comp] = []

    scene_stats = {}
    for scene in scene_categories:
        scene_stats[scene] = []

    total_num = 0
    for image_name, anno in tqdm(element_anno.items()):
        # read source image
        image_file = os.path.join(image_dir, image_name)
        assert os.path.exists(image_file), image_file
        src = cv2.imread(os.path.join(image_dir, image_name))
        im_h, im_w, _ = src.shape
        total_num += 1
        # store images to different subfolders according to mean score
        im_scores = score_anno[image_name]['scores']
        mean_score = float(sum(im_scores)) / len(im_scores)
        im_level = math.floor(mean_score) - 1 if mean_score < 5 else 3
        im_level = score_levels[im_level]
        score_stats[im_level].append(image_name)
        subfolder = os.path.join(score_dir, im_level)
        os.makedirs(subfolder, exist_ok=True)
        text = 'score:{:.1f}'.format(mean_score)
        dst = src.copy()
        cv2.putText(dst, text, (20, 70), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 255), 3)
        cv2.imwrite(os.path.join(subfolder, image_name), dst)
        # readout scene annotation
        per_scene = scene_anno[image_name]
        assert per_scene in scene_categories, per_scene
        os.makedirs(os.path.join(scene_dir, per_scene), exist_ok=True)
        cv2.imwrite(os.path.join(scene_dir, per_scene, image_name), src)
        scene_stats[per_scene].append(image_name)
        # visualize composition elements annotation
        for comp, element in anno.items():
            element_stats[comp].append(image_name)
            dst = draw_element_on_image(src.copy(), comp, element)
            subpath = os.path.join(element_dir, comp)
            if not os.path.exists(subpath):
                os.makedirs(subpath)
            cv2.imwrite(os.path.join(subpath, image_name), dst)

    # show dataset statistical information
    element_stats = sorted(element_stats.items(), key=lambda d: len(d[1]), reverse=True)
    scene_stats   = sorted(scene_stats.items(), key=lambda d: len(d[1]), reverse=True)

    with open(os.path.join(opt.save_folder, 'statistics.txt'), 'w') as f:
        f.write('Composition Score\n')
        print('Composition score')
        for level, img_list in score_stats.items():
            ss = '{}: {} images, {:.1%}'.format(level, len(img_list), len(img_list) / total_num)
            f.write(ss + '\n')
            print(ss)

        print('\nComposition Element')
        f.write('\nComposition Element\n')
        for comp, img_list in element_stats:
            ss = '{}: {} images, {:.1%}'.format(comp, len(img_list), len(img_list) / total_num)
            f.write(ss + '\n')
            print(ss)

        print('\nScene Category')
        f.write('\nScene Category\n')
        for scene, img_list in scene_stats:
            ss = '{}: {} images, {:.1%}'.format(scene, len(img_list), len(img_list) / total_num)
            f.write(ss + '\n')
            print(ss)

        ss = 'Total number of images: {}'.format(total_num)
        print(ss + '\n')
        f.write(ss)
