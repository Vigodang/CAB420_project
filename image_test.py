from libs.datasets_keras import load_lines, load_img
from libs.config import LABELMAP_RGB
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _first_scene_chip(scene, chip_dir, dataset='data'):
    path = Path(dataset) / chip_dir
    matches = sorted(path.glob(f'{scene}-*.png'))
    if not matches:
        raise FileNotFoundError(f'No chip files found for scene "{scene}" in {chip_dir}')
    return matches[len(matches) // 2]


train_lines = load_lines('data/train.txt')
train_scene = train_lines[5]

first_chip = _first_scene_chip(train_scene, 'image-chips')
img = load_img(str(first_chip))
eleva = load_img(str(Path('data/eleva-chips') / first_chip.name))
label = load_img(str(Path('data/label-chips') / first_chip.name))

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title('Image chip')
axes[0].axis('off')

axes[1].imshow(eleva if eleva.ndim == 2 else eleva[..., 0], cmap='terrain')
axes[1].set_title('Elevation chip')
axes[1].axis('off')

label_class = label[:, :, 0] if label.ndim == 3 and label.shape[2] > 1 else label
label_colors = np.array([LABELMAP_RGB[i + 1] for i in range(len(LABELMAP_RGB) - 1)], dtype=np.uint8)
label_img = label_colors[label_class]
axes[2].imshow(label_img)
axes[2].set_title('Label chip')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('first_train_chip_preview.png', dpi=150)