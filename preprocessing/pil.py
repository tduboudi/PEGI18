import os
from PIL import Image

_CROP = 0.7
_DATASET_DIR = "train_cropped/scary"
_OUTPUT_DIR  = "cropped/scary"

def _get_filenames(dataset_dir):
  photo_filenames = []
  for filename in os.listdir(dataset_dir):
    path = os.path.join(dataset_dir, filename)
    photo_filenames.append(path)

  return photo_filenames

def _convert_dataset(filenames):
  for filename in filenames:
    image = Image.open(filename)

    height, width = image.size

    wc = width*_CROP
    hc = height*_CROP

    wnc = width*(1. - _CROP)
    hnc = height*(1. - _CROP)

    imageTL = image.crop((0, 0, hc, wc))
    imageTR = image.crop((0, wnc, hc, width))
    imageBL = image.crop((hnc, 0, height, wc))
    imageBR = image.crop((hnc, wnc, height, width))

    imageTL.show()
    imageTR.show()
    imageBL.show()
    imageBR.show()

    imageTL.save(os.path.join(_OUTPUT_DIR, str(os.path.basename(filename)) + "_tl_" + ".jpg"))
    imageTR.save(os.path.join(_OUTPUT_DIR, str(os.path.basename(filename)) + "_tr_" + ".jpg"))
    imageBL.save(os.path.join(_OUTPUT_DIR, str(os.path.basename(filename)) + "_bl_" + ".jpg"))
    imageBR.save(os.path.join(_OUTPUT_DIR, str(os.path.basename(filename)) + "_br_" + ".jpg"))

def run(dataset_dir):
  filenames = _get_filenames(dataset_dir)
  _convert_dataset(filenames)

run(_DATASET_DIR)
