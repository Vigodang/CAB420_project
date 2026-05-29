import os
import random

LABELS = ['BUILDING', 'CLUTTER', 'VEGETATION', 'WATER', 'GROUND', 'CAR']

# Class to color (BGR)
LABELMAP = {
    0: (255, 0, 255),
    1: (75, 25, 230),
    2: (180, 30, 145),
    3: (75, 180, 60),
    4: (48, 130, 245),
    5: (255, 255, 255),
    6: (200, 130, 0),
}

# Color (BGR) to class
INV_LABELMAP = {
    (255, 0, 255): 0,
    (75, 25, 230): 1,
    (180, 30, 145): 2,
    (75, 180, 60): 3,
    (48, 130, 245): 4,
    (255, 255, 255): 5,
    (200, 130, 0): 6,
}

LABELMAP_RGB = {k: (v[2], v[1], v[0]) for k, v in LABELMAP.items()}
INV_LABELMAP_RGB = {v: k for k, v in LABELMAP_RGB.items()}

# Whether the model and data pipeline should use both RGB imagery and elevation.
# When False, only RGB image channels are used.
USE_ELEVATION = True
IMAGE_CHANNELS = 3
ELEVATION_CHANNELS = 1
SIZE = 128

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SPLIT_SEED = 42
SPLIT_SHUFFLE = True
DEFAULT_INDEX_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'index.csv'))


def input_channels(use_elevation=None):
    if use_elevation is None:
        use_elevation = USE_ELEVATION
    return IMAGE_CHANNELS + ELEVATION_CHANNELS if use_elevation else IMAGE_CHANNELS


def load_scene_ids_from_index(index_csv_path):
    index_csv_path = os.path.abspath(index_csv_path)
    if not os.path.exists(index_csv_path):
        raise FileNotFoundError(f'Index CSV not found: {index_csv_path}')
    scene_ids = []
    with open(index_csv_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 2:
                continue
            scene_ids.append(parts[1])
    return scene_ids


def split_scene_ids(scene_ids, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
                    test_ratio=TEST_RATIO, seed=SPLIT_SEED, shuffle=SPLIT_SHUFFLE):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f'Split ratios must sum to 1.0, got {total}')

    scene_ids = list(scene_ids)
    if shuffle:
        random.Random(seed).shuffle(scene_ids)

    n_total = len(scene_ids)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))
    if n_train + n_val > n_total:
        n_val = max(0, n_total - n_train)

    train_ids = scene_ids[:n_train]
    val_ids = scene_ids[n_train:n_train + n_val]
    test_ids = scene_ids[n_train + n_val:]
    return train_ids, val_ids, test_ids


def split_scene_ids_from_index(index_csv_path=DEFAULT_INDEX_CSV,
                                train_ratio=TRAIN_RATIO,
                                val_ratio=VAL_RATIO,
                                test_ratio=TEST_RATIO,
                                seed=SPLIT_SEED,
                                shuffle=SPLIT_SHUFFLE):
    scene_ids = load_scene_ids_from_index(index_csv_path)
    return split_scene_ids(scene_ids,
                           train_ratio=train_ratio,
                           val_ratio=val_ratio,
                           test_ratio=test_ratio,
                           seed=seed,
                           shuffle=shuffle)


def get_dataset_split_ids(dataset_dir, index_filename='index.csv', **kwargs):
    return split_scene_ids_from_index(os.path.join(dataset_dir, index_filename), **kwargs)


try:
    train_ids, val_ids, test_ids = split_scene_ids_from_index()
except FileNotFoundError:
    train_ids, val_ids, test_ids = [], [], []
