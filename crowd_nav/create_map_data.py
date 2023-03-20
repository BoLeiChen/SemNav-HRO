import os
import tqdm
import argparse
import multiprocessing as mp
from poni.default import get_cfg
from poni.dataset_1 import SemanticMapDataset
import matplotlib.pyplot as plt
import cv2
import numpy

# assert 'ACTIVE_DATASET' in os.environ
# ACTIVE_DATASET = os.environ['ACTIVE_DATASET']
ACTIVE_DATASET = "mp3d"
DATASET = ACTIVE_DATASET
OUTPUT_MAP_SIZE = 24.0
MASKING_MODE = "spath"
MASKING_SHAPE = "square"

SEED = 123
DATA_ROOT = "/home/rx/RelationalGraphLearning/crowd_nav/semantic_maps"
FMM_DISTS_SAVED_ROOT = "/home/rx/RelationalGraphLearning/crowd_nav/fmm_dists_123"
NUM_SAMPLES = {'train': 440, 'val': 105}
SAVE_ROOT = "/home/rx/RelationalGraphLearning/crowd_nav/precomputed_dataset_mp3d_24.0_123_spath"

classes_SSCNav = ['wall', 'floor', 'chair', 'door', 'table', 'picture', 'cabinet', 'cushion', 'window', 'sofa',
                  'bed', 'curtain', 'chest_of_drawers', 'plant', 'sink', 'stairs', 'ceiling', 'toilet', 'stool',
                  'towel', 'mirror', 'tv_monitor', 'shower', 'column', 'bathtub', 'counter', 'fireplace', 'lighting',
                  'beam', 'railing', 'shelving', 'blinds', 'gym_equipment', 'seating', 'board_panel', 'furniture',
                  'appliances', 'clothes', 'objects', 'misc']

classes_PONI = ["floor", "wall", "chair", "table", "picture", "cabinet", "cushion", "sofa", "bed", "chest_of_drawers",
                "plant", "sink", "toilet", "stool", "towel", "tv_monitor", "shower", "bathtub", "counter", "fireplace",
                "gym_equipment", "seating", "clothes"]

def get_approx(img, contours, length_p=0.1):
    """获取逼近多边形

    :param img: 处理图片
    :param contour: 连通域
    :param length_p: 逼近长度百分比
    """
    img_adp = img.copy()
    print(len(contours))
    all_approx = []
    cnt = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > 20:
            # 逼近长度计算
            epsilon = length_p * cv2.arcLength(contours[i], True)
            # 获取逼近多边形
            approx = cv2.approxPolyDP(contours[i], epsilon, True)
            print(approx)
            cnt += numpy.size(approx)
            # 绘制显示多边形
            all_approx = all_approx + [approx]
            cv2.drawContours(img_adp, [approx], 0, (0, 0, 255), 2)
    print("point size:", cnt)
    print(all_approx)
    cv2.imshow("result %.5f" % length_p, img_adp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def precompute_dataset_for_map(kwargs):
    cfg = kwargs["cfg"]
    split = kwargs["split"]
    name = kwargs["name"]

    dataset = SemanticMapDataset(
        cfg.DATASET, split=split, scf_name=name, seed=SEED
    )
    name, semmap, fmm_dists, map_xyz_info, nav_space, nav_locs = dataset.get_item_by_name(name)
    #print("semmap", nav_space.shape)
    print(nav_space)
    channel = semmap[1, ...].astype(numpy.uint8) * 255  # free:0, Occ:255
    vis = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
    img_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(img_gray, 50, 200)
    minLineLength = 15  # 最低线段的长度，小于这个值的线段被抛弃
    maxLineGap = 10  # 线段中点与点之间连接起来的最大距离，在此范围内才被认为是单行
    #n,1,4包含每条线段的两个顶点
    lines = cv2.HoughLinesP(edges, 1, numpy.pi / 180, 15, minLineLength=minLineLength, maxLineGap=maxLineGap)
    #print(lines)
    print("line size:", len(lines))
    for i in range(len(lines)):
        x_1, y_1, x_2, y_2 = lines[i][0]
        cv2.line(vis, (x_1, y_1), (x_2, y_2), (0, 255, 0), 2)
    cv2.imshow("result", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    channel = semmap[2, ...].astype(numpy.uint8) * 255  # free:0, Occ:255
    plt.subplot(1, 3, 1)
    plt.imshow(channel)
    vis = cv2.cvtColor(channel, cv2.COLOR_GRAY2RGB)
    img_gray = cv2.cvtColor(vis, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    get_approx(vis, contours, 0.05)
    '''
    # 4.获取不同尺度的逼近多边形
    get_approx(vis, contours[0], 0.15)
    get_approx(vis, contours[0], 0.09)
    get_approx(vis, contours[0], 0.05)
    get_approx(vis, contours[0], 0.02)
    get_approx(vis, contours[0], 0.002)
    '''

    local_map_1 = dataset.visualize_map(semmap[0:23, ...], dataset="mp3d")
    print(local_map_1.shape)
    plt.subplot(1, 3, 2)
    plt.imshow(local_map_1)

    plt.subplot(1, 3, 3)
    plt.imshow(fmm_dists[2])
    plt.show()


def precompute_dataset(args):
    cfg = get_cfg()
    cfg.defrost()
    cfg.SEED = SEED
    cfg.DATASET.dset_name = DATASET
    cfg.DATASET.root = DATA_ROOT
    cfg.DATASET.output_map_size = OUTPUT_MAP_SIZE
    cfg.DATASET.fmm_dists_saved_root = FMM_DISTS_SAVED_ROOT
    cfg.DATASET.masking_mode = MASKING_MODE
    cfg.DATASET.masking_shape = MASKING_SHAPE
    cfg.DATASET.visibility_size = 3.0  # m
    cfg.freeze()

    os.makedirs(SAVE_ROOT, exist_ok=True)

    os.makedirs(os.path.join(SAVE_ROOT, args.split), exist_ok=True)
    dataset = SemanticMapDataset(cfg.DATASET, split=args.split)
    n_maps = len(dataset)
    print('Maps', n_maps)
    n_samples_per_map = (NUM_SAMPLES[args.split] // n_maps)
    print(n_samples_per_map)
    print(n_maps)

    if args.map_id != -1:
        map_names = [dataset.names[args.map_id]]
    elif args.map_id_range is not None:
        assert len(args.map_id_range) == 2
        map_names = [
            dataset.names[i]
            for i in range(args.map_id_range[0], args.map_id_range[1] + 1)
        ]
    else:
        map_names = dataset.names

    pool = mp.Pool(processes=args.num_workers)
    inputs = []
    for name in map_names:
        kwargs = {
            "cfg": cfg,
            "split": args.split,
            "name": name,
            "n_samples_per_map": n_samples_per_map,
            "save_root": f'{SAVE_ROOT}/{args.split}',
        }
        inputs.append(kwargs)

    with tqdm.tqdm(total=len(inputs)) as pbar:
        for _ in pool.imap_unordered(precompute_dataset_for_map, inputs):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-id', type=int, default=-1)
    parser.add_argument('--map-id-range', type=int, nargs="+", default=None)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--num-workers', type=int, default=1)
    args = parser.parse_args()

    # Both map-id and map-id-range should not be enabled simultaneously
    assert (args.map_id == -1) or (args.map_id_range is None)

    precompute_dataset(args)