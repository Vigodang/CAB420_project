from PIL import Image
import numpy as np
import math
from keras import models
import os

from libs.config import get_dataset_split_ids, LABELMAP_RGB, USE_ELEVATION

def category2mask(img):
    """ Convert a category image to color mask """
    if len(img) == 3:
        if img.shape[2] == 3:
            img = img[:, :, 0]

    mask = np.zeros(img.shape[:2] + (3, ), dtype='uint8')

    for category, mask_color in LABELMAP_RGB.items():
        locs = np.where(img == category)
        mask[locs] = mask_color

    return mask

def chips_from_image(img, size=300):
    shape = img.shape

    chips = []
    for x in range(0, shape[1], size):
        for y in range(0, shape[0], size):
            chip = img[y:y+size, x:x+size]
            y_pad = size - chip.shape[0]
            x_pad = size - chip.shape[1]
            pad_width = [(0, y_pad), (0, x_pad)] + [(0, 0)] * (chip.ndim - 2)
            chip = np.pad(chip, pad_width, mode='constant')
            chips.append((chip, x, y))
    return chips

def run_inference_on_file(imagefile, predsfile, model, size=300, use_elevation=False, elevafile=None):
    with Image.open(imagefile).convert('RGB') as img:
        nimg = np.array(img)
        shape = nimg.shape

    elevation_chips = None
    if use_elevation:
        if elevafile is None:
            raise ValueError('elevafile must be provided when use_elevation=True')
        if not os.path.exists(elevafile):
            raise FileNotFoundError(f'Elevation file not found: {elevafile}')

        with Image.open(elevafile) as elev_img:
            ne = np.array(elev_img.convert('L'))
        elevation_chips = chips_from_image(ne, size=size)

    image_chips = chips_from_image(nimg, size=size)

    if use_elevation:
        if len(image_chips) != len(elevation_chips):
            raise ValueError('Image and elevation chips count mismatch')

        chips = []
        for (img_chip, xi, yi), (eleva_chip, xe, ye) in zip(image_chips, elevation_chips):
            if xi != xe or yi != ye:
                raise ValueError('Mismatched chip coordinates between image and elevation')
            if eleva_chip.ndim == 2:
                eleva_chip = eleva_chip[..., np.newaxis]
            chips.append((np.concatenate([img_chip, eleva_chip], axis=-1), xi, yi))
    else:
        chips = image_chips

    chips = [(chip, xi, yi) for chip, xi, yi in chips if chip.sum() > 0]
    if not chips:
        raise ValueError(f'No valid chips generated for {imagefile}')

    prediction = np.zeros(shape[:2], dtype='uint8')
    chip_preds = model.predict(np.array([chip for chip, _, _ in chips]), verbose=True)

    for (chip, x, y), pred in zip(chips, chip_preds):
        category_chip = np.argmax(pred, axis=-1) + 1
        section = prediction[y:y+size, x:x+size].shape
        prediction[y:y+size, x:x+size] = category_chip[:section[0], :section[1]]

    mask = category2mask(prediction)
    Image.fromarray(mask).save(predsfile)

def run_inference(dataset, model=None, model_path=None, basedir='predictions', use_elevation=None):
    if not os.path.isdir(basedir):
        os.mkdir(basedir)
    if model is None and model_path is None:
        raise Exception("model or model_path required")

    if model is None:
        model = models.load_model(model_path)

    if use_elevation is None:
        use_elevation = USE_ELEVATION

    train_ids, val_ids, test_ids = get_dataset_split_ids(dataset)
    for scene in train_ids + val_ids + test_ids:
        imagefile = f'{dataset}/images/{scene}-ortho.tif'
        predsfile = os.path.join(basedir, f'{scene}-prediction.png')
        elevafile = f'{dataset}/elevations/{scene}-elev.tif' if use_elevation else None

        if not os.path.exists(imagefile):
            continue
        if use_elevation and not os.path.exists(elevafile):
            raise FileNotFoundError(f'Elevation file required by config but not found: {elevafile}')

        print(f'running inference on image {imagefile}.')
        run_inference_on_file(imagefile, predsfile, model, use_elevation=use_elevation, elevafile=elevafile)
