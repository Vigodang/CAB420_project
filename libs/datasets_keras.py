from keras import Sequential
from keras.layers import Layer, RandomFlip, RandomRotation
from keras.utils import Sequence
from PIL import Image

from pathlib import Path
import numpy as np
import random

BASE_AUGMENTATION = Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.5, interpolation='bilinear', fill_mode='reflect')
])

def load_dataset(dataset, bs, use_elevation=True, aug=BASE_AUGMENTATION):
    train_image_files, train_eleva_files = _expand_split_chips(dataset, 'train.txt', use_elevation)
    valid_image_files, valid_eleva_files = _expand_split_chips(dataset, 'valid.txt', use_elevation)

    image_augmenter = _resolve_augmentation(aug)
    label_augmenter = _build_label_augmentation(image_augmenter)
    
    train_seq = SegmentationSequence(
        dataset,
        train_image_files,
        train_eleva_files,
        image_augmenter,
        label_augmenter,
        bs,
        use_elevation
    )
    
    valid_seq = SegmentationSequence(
        dataset,
        valid_image_files,
        valid_eleva_files,
        None,  # don't augment validation set
        None,
        bs,
        use_elevation
    )
    
    return train_seq, valid_seq


def _resolve_augmentation(aug):
    if aug is False or aug == [] or aug is None:
        return None

    if isinstance(aug, Sequential):
        return aug

    raise ValueError('aug must be None, False, an empty list, or a keras.Sequential instance')


def _is_label_safe_layer(layer):
    name = layer.__class__.__name__
    return name in {
        'RandomFlip',
        'RandomRotation',
        'RandomTranslation',
        'RandomZoom',
        'RandomCrop',
        'CenterCrop'
    }


def _clone_layer_for_labels(layer):
    config = layer.get_config()
    if 'interpolation' in config:
        config['interpolation'] = 'nearest'
    return layer.__class__.from_config(config)


def _build_label_augmentation(image_augmenter):
    if image_augmenter is None:
        return None

    safe_layers = []
    for layer in image_augmenter.layers:
        if _is_label_safe_layer(layer):
            safe_layers.append(_clone_layer_for_labels(layer))

    return Sequential(safe_layers) if safe_layers else None


def _resolve_split_file(dataset, filename):
    path = Path(dataset) / filename
    if path.exists():
        return path
    if filename == 'valid.txt':
        alt = Path(dataset) / 'val.txt'
        if alt.exists():
            return alt
    raise FileNotFoundError(f'Split file not found: {dataset}/{filename}')


def _expand_split_chips(dataset, split_file, use_elevation=True):
    split_path = _resolve_split_file(dataset, split_file)
    scenes = load_lines(str(split_path))
    image_files = []
    eleva_files = []

    for scene in scenes:
        image_matches = sorted(Path(dataset, 'image-chips').glob(f'{scene}-*.png'))

        if not image_matches:
            raise FileNotFoundError(f'No image chips found for scene "{scene}" in {dataset}/image-chips')

        image_files.extend(str(p) for p in image_matches)

        if use_elevation:
            eleva_matches = sorted(Path(dataset, 'eleva-chips').glob(f'{scene}-*.png'))
            if len(image_matches) != len(eleva_matches):
                raise ValueError(
                    f'Mismatched chip counts for scene "{scene}" in {dataset}: '
                    f'{len(image_matches)} image chips vs {len(eleva_matches)} elevation chips'
                )
            eleva_files.extend(str(p) for p in eleva_matches)

    return image_files, eleva_files

def load_lines(fname):
    with open(fname, 'r') as f:
        return [l.strip() for l in f.readlines()]

def load_img(fname):
    return np.array(Image.open(fname))

def mask_to_classes(mask):
    """Return integer class map suitable for sparse categorical loss.

    Ensures a single-channel integer map with values 0..5 and shape HxWx1.
    """
    # If mask has channel dim, assume class ids are in channel 0
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask.astype('int32')[..., np.newaxis]

class SegmentationSequence(Sequence):
    def __init__(self, dataset, image_files, eleva_files, image_augmenter, label_augmenter, bs, use_elevation=True):
        if use_elevation:
            assert len(image_files) == len(eleva_files), 'image and eleva chip counts must match'
            self.samples = list(zip(image_files, eleva_files))
        else:
            assert not eleva_files, 'Elevation files must be empty when use_elevation is False'
            self.samples = list(image_files)

        self.use_elevation = use_elevation
        self.label_path = f'{dataset}/label-chips'
        self.image_path = f'{dataset}/image-chips'
        self.eleva_path = f'{dataset}/eleva-chips'
        random.shuffle(self.samples)

        self.image_augmenter = image_augmenter
        self.label_augmenter = label_augmenter
        self.bs = bs

    def __len__(self):
        return int(np.ceil(len(self.samples) / float(self.bs)))

    def __getitem__(self, idx):
        batch = self.samples[idx*self.bs:(idx+1)*self.bs]
        if self.use_elevation:
            image_files = [img for img, _ in batch]
            eleva_files = [ele for _, ele in batch]
        else:
            image_files = batch
            eleva_files = []

        label_files = [fname.replace(self.image_path, self.label_path) for fname in image_files]

        images = [load_img(fname) for fname in image_files]
        if self.use_elevation:
            elevas = [load_img(fname) for fname in eleva_files]
            images = np.array(images)
            elevas = np.array([el[..., np.newaxis] if el.ndim == 2 else el[..., :1] for el in elevas])
        else:
            images = np.array(images)
            elevas = None

        labels = np.array([mask_to_classes(load_img(fname)) for fname in label_files])

        if self.image_augmenter is not None and self.label_augmenter is not None:
            seed = random.randint(0, 2**31 - 1)
            if self.use_elevation:
                combined = np.concatenate([images, elevas], axis=-1)
                combined = self.image_augmenter(combined, training=True, seed=seed).numpy()
                images = combined[..., :3]
                elevas = combined[..., 3:]
            else:
                images = self.image_augmenter(images, training=True, seed=seed).numpy()
            labels = self.label_augmenter(labels, training=True, seed=seed).numpy()

        if self.use_elevation:
            return (images, elevas), labels
        return images, labels

    def _stack_elevation(self, image, eleva):
        if eleva.ndim == 2:
            eleva = eleva[..., np.newaxis]
        elif eleva.ndim == 3 and eleva.shape[2] > 1:
            eleva = eleva[..., :1]

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

        return np.concatenate([image, eleva], axis=-1)

    def on_epoch_end(self):
        random.shuffle(self.samples)
