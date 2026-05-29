import cv2
import os
import numpy as np

from libs.config import split_scene_ids_from_index, LABELMAP, INV_LABELMAP, SIZE

size   = SIZE
stride = SIZE

def color2class(orthochip, img):
    ret = np.zeros((img.shape[0], img.shape[1]), dtype='uint8')
    ret = np.dstack([ret, ret, ret])
    colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)

    # Skip any chips that would contain magenta (IGNORE) pixels
    seen_colors = set( [tuple(color) for color in colors] )
    IGNORE_COLOR = LABELMAP[0]
    if IGNORE_COLOR in seen_colors:
        return None, None

    for color in colors:
        locs = np.where( (img[:, :, 0] == color[0]) & (img[:, :, 1] == color[1]) & (img[:, :, 2] == color[2]) )
        ret[ locs[0], locs[1], : ] = INV_LABELMAP[ tuple(color) ] - 1

    return orthochip, ret

def image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile, windowx=size, windowy=size, stridex=stride, stridey=stride):

    ortho = cv2.imread(orthofile)
    label = cv2.imread(labelfile)

    # Not using elevation in the sample - but useful to incorporate it ;)
    eleva = cv2.imread(elevafile, -1)

    assert(ortho.shape[0] == label.shape[0])
    assert(ortho.shape[1] == label.shape[1])

    shape = ortho.shape

    xsize = shape[1]
    ysize = shape[0]
    print(f"converting {dataset} image {orthofile} {xsize}x{ysize} to chips ...")

    counter = 0

    for xi in range(0, shape[1] - windowx, stridex):
        for yi in range(0, shape[0] - windowy, stridey):

            orthochip = ortho[yi:yi+windowy, xi:xi+windowx, :]
            labelchip = label[yi:yi+windowy, xi:xi+windowx, :]
            elevachip = eleva[yi:yi+windowy, xi:xi+windowx]

            orthochip, classchip = color2class(orthochip, labelchip)

            if classchip is None:
                continue

            orthochip_filename = os.path.join(prefix, 'image-chips', scene + '-' + str(counter).zfill(6) + '.png')
            labelchip_filename = os.path.join(prefix, 'label-chips', scene + '-' + str(counter).zfill(6) + '.png')
            elevachip_filename = os.path.join(prefix, 'eleva-chips', scene + '-' + str(counter).zfill(6) + '.png')

            # We no longer write individual chip filenames to the dataset files here.
            # The dataset txt files should list the overall scene/image identifiers instead.

            cv2.imwrite(orthochip_filename, orthochip)
            cv2.imwrite(labelchip_filename, classchip)
            cv2.imwrite(elevachip_filename, elevachip)
            counter += 1


def get_split(scene, train_set, val_set, test_set):
    if scene in train_set:
        return 'train.txt'
    if scene in val_set:
        return 'valid.txt'
    if scene in test_set:
        return 'test.txt'
    return None


def run(prefix):
    open(prefix + '/train.txt', mode='w').close()
    open(prefix + '/valid.txt', mode='w').close()
    open(prefix + '/test.txt', mode='w').close()

    if not os.path.exists(os.path.join(prefix, 'image-chips')):
        os.mkdir(os.path.join(prefix, 'image-chips'))
    if not os.path.exists(os.path.join(prefix, 'label-chips')):
        os.mkdir(os.path.join(prefix, 'label-chips'))
    if not os.path.exists(os.path.join(prefix, 'eleva-chips')):
        os.mkdir(os.path.join(prefix, 'eleva-chips'))

    train_ids, val_ids, test_ids = split_scene_ids_from_index(os.path.join(prefix, 'index.csv'))
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    print(f'split scenes: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}')

    lines = [line for line in open(f'{prefix}/index.csv')]
    num_images = len(lines)
    print(f"converting {num_images} images to chips - this may take a few minutes but only needs to be done once.")

    for lineno, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        if len(parts) < 2:
            continue
        scene = parts[1]
        dataset = get_split(scene, train_set, val_set, test_set)
        if dataset is None:
            continue

        # Record the overall scene identifier in the dataset file
        with open(os.path.join(prefix, dataset), mode='a') as fd:
            fd.write(scene + '\n')

        if dataset == 'test.txt':
            print(f"not converting test image {scene} to chips, it will be used for inference.")
            continue

        orthofile = os.path.join(prefix, 'images', scene + '-ortho.tif')
        elevafile = os.path.join(prefix, 'elevations', scene + '-elev.tif')
        labelfile = os.path.join(prefix, 'labels', scene + '-label.png')

        if os.path.exists(orthofile) and os.path.exists(labelfile) and os.path.exists(elevafile):
            image2tile(prefix, scene, dataset, orthofile, elevafile, labelfile)
