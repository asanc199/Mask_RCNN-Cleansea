import os
import sys
from numpy import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import json
from PIL import Image, ImageDraw
from tensorflow.python.framework.versions import VERSION as __version__
import tensorflow as tf
from imgaug import augmenters as iaa
import warnings
warnings.filterwarnings(action='ignore')

#Cambiamos el Directorio al propio de MASK_RCNN
ROOT_DIR = 'D:/Cleansea/Mask_RCNN-cleansea'
#ROOT_DIR = '/home/saflex/projecto_cleansea/Mask_RCNN/Mask_RCNN-master'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist'

# Import mrcnn libraries
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
# Directorio perteneciente a MASK-RCNN
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Ruta al archivo de pesos
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Descargamos los Pesos Entrenados de COCO
if not os.path.exists(COCO_WEIGHTS_PATH):
    utils.download_trained_weights(COCO_WEIGHTS_PATH)


############################################################
#  Configuracion
############################################################

class CleanSeaConfig(Config):
    """
    Configuracion para el entrenamiento con CleanSea Dataset.
    """

    # Nombre de la configuracion
    NAME = "debris"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Numero de clases + el background
    NUM_CLASSES = 1 + 19  # Cleansea tiene 19 clases

    # Salta las detecciones con <50% de seguridad
    DETECTION_MIN_CONFIDENCE = 0.5

    #Learning Rate Modificado
    LEARNING_RATE = 0.001
    

config= CleanSeaConfig()
config.display()

def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

############################################################
#  Dataset
############################################################
class CleanSeaDataset(utils.Dataset):
    def load_data(self, dataset_dir, subset):
        # Train or validation dataset?
        assert subset in ["train_coco", "test_coco"]
        dataset_dir = os.path.join(dataset_dir, subset)
        print(dataset_dir)

        # Cargamos el archivo json
        annotation_json = os.path.join(dataset_dir,"annotations.json")
        json_file = open(annotation_json)
        coco_json = json.load(json_file)
        json_file.close()
        print("\nAnotaciones Cargadas\n")

        # Añadimos los nombres de las clases usando el metodo de utils.Dataset
        source_name = "coco_like"
        for category in coco_json['categories']:
            class_id = category['id']
            class_name = category['name']
            if class_id < 1:
                print('Error: Class id for "{}" reserved for the background'.format(class_name))
            else:
                self.add_class(source_name, class_id, class_name)
        print("Nombres Añadidos \n")

        # Almacenamos las anotaciones
        annotations = {}
        for annotation in coco_json['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations:
                annotations[image_id] = []
            annotations[image_id].append(annotation)
        print("Anotaciones Almacenadas\n")

        # Almacenamos las imagenes y las añadimos al dataset
        seen_images = {}
        for image in coco_json['images']:
            image_id = image['id']
            if image_id in seen_images:
                print("Warning: Skipping duplicate image id: {}".format(image))
            else:
                seen_images[image_id] = image
                try:
                    image_file_name = image['file_name']
                    image_width = image['width']
                    image_height = image['height']
                except KeyError as key:
                    print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))
                
                image_path = os.path.join(dataset_dir, image_file_name)
                image_annotations = annotations[image_id]
                
                # Añadimos la imagen usando el metodo de utils.Dataset
                self.add_image(
                    source=source_name,
                    image_id=image_id,
                    path=image_path,
                    width=image_width,
                    height=image_height,
                    annotations=image_annotations
                )
        print("Imagenes añadidas al Dataset\n")

    def load_mask(self, image_id):
        """ Carga la mascara de instancia para la imagen dada
        MaskRCNN espera mascaras en forma de mapa de bits (altura, anchura e instancias)
        Argumentos:
            image_id: El ID de la imagen a la que vamos a cargar la mascara
        Salida:
            masks: Una cadena booleana con estructura (altura, anchya y la cuenta de instancias) con una mascara por instancia
            class_ids: Una cadena de 1 dimension de clase ID de la instancia de la mascara """
        image_info = self.image_info[image_id]
        annotations = image_info['annotations']
        instance_masks = []
        class_ids = []
        
        for annotation in annotations:
            class_id = annotation['category_id']
            mask = Image.new('1', (image_info['width'], image_info['height']))
            mask_draw = ImageDraw.ImageDraw(mask, '1')
            for segmentation in annotation['segmentation']:
                mask_draw.polygon(segmentation, fill=1)
                bool_array = np.array(mask) > 0
                instance_masks.append(bool_array)
                class_ids.append(class_id)

        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

"""Train the model."""
# Training dataset.
dataset_train = CleanSeaDataset()
print("Configuracion para train cargada\n")
dataset_train.load_data("D:/Cleansea/cleansea_dataset/CocoFormatDataset","train_coco")
print("Dataset Inicializado Correctamente\n")
dataset_train.prepare()
print("Preparacion del Dataset Completada\n")

# Validation dataset
dataset_test = CleanSeaDataset()
print("Configuracion para test cargada\n")
dataset_test.load_data("D:/Cleansea/cleansea_dataset/CocoFormatDataset", "test_coco")
print("Dataset Inicializado Correctamente\n")
dataset_test.prepare()
print("Preparacion del Dataset Completada\n")

# Load and display random samples
print("Mostrando Imagenes aleatorias...\n")

image_ids = np.random.choice(dataset_train.image_ids, 4)
for image_id in image_ids:
    image = dataset_train.load_image(image_id)
    mask, class_ids = dataset_train.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset_train.class_names)

print("Inicializing model for training...\n")
model=modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# Which weights to start with?

init_with = "coco"  # imagenet, coco, or last
if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_WEIGHTS_PATH, by_name=True,
                    exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

last_path="D:/Cleansea/Mask_RCNN-cleansea/logs/mask_rcnn_debris_weights1000DA5Heads.h5"
model.load_weights(last_path, by_name=True)


############################################################
#  Training
############################################################
# ===============================================================================
# Aumentado de Datos
# ===============================================================================

seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    # Small gaussian blur with random sigma between 0 and 0.5.
    # But we only blur about 50% of all images.
    iaa.Sometimes(
        0.5,
        iaa.GaussianBlur(sigma=(0, 0.5))
    ),
    # Strengthen or weaken the contrast in each image.
    iaa.LinearContrast((0.75, 1.5)),
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
    iaa.Multiply((0.8, 1.2), per_channel=0.2),
    # Apply affine transformations to each image.
    # Scale/zoom them, translate/move them, rotate them and shear them.
    iaa.Affine(
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-25, 25),
        shear=(-8, 8)
    )
], random_order=True) # apply augmenters in random order

"""
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
])
"""

# ===============================================================================

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.

print("Entrenando Heads...\n")
model.train(dataset_train, dataset_test, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads',augmentation=seq)

# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
print("Entrenando extensivamente...\n")
model.train(dataset_train, dataset_test, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=1000, 
            layers="all",augmentation=seq)

# Save weights
# Typically not needed because callbacks save after every epoch
# Uncomment to save manually
print("Guardando Pesos...\n")
model_path = os.path.join(MODEL_DIR, "mask_rcnn_debris_weights.h5")
model.keras_model.save_weights(model_path)
print("Pesos Guardados en mask_rcnn_debris_weights.h5")

############################################################
#  Evaluacion
############################################################
class InferenceConfig(CleanSeaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    USE_MINI_MASK = False

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

image_ids = dataset_test.image_ids
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_test, inference_config,
                               image_id)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))