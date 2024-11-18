#MRCNN COCO PROJECT

from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from mrcnn.utils import Dataset
from PIL import Image
import os

# PATH TO fichier_instances
fichier_instances = 'C:/Users/oassa/Downloads/PROJET IMANE/annotations/instances_train2017.json'

# let's load instance annotations
coco = COCO(fichier_instances)

# There are many categories, we are searching for HORSES
categories = coco.loadCats(coco.getCatIds())
horse_names = [horse['name'] for horse in categories]
print("Catégories disponibles :", horse_names)

# As we have many IDs, let's extract only 'horse' IDs
horse_ids = coco.getCatIds(catNms=['horse'])
image_ids = coco.getImgIds(catIds=horse_ids)

# loading the images
images = coco.loadImgs(image_ids)
print(f"Nombre d'images contenant des chevaux : {len(images)}")

if len(image_ids) > 0:
    image_id = image_ids[0]  # Utiliser la première image contenant des chevaux
    image_info = coco.loadImgs([image_id])[0]
    image_path = f'C:/Users/oassa/Downloads/PROJET IMANE/train2017/train2017/{image_info["file_name"]}'

    # Vérification si l'image existe
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"L'image {image_path} n'a pas pu être chargée.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Charger les annotations pour cette image
        annotation_ids = coco.getAnnIds(imgIds=[image_id], catIds=horse_ids)
        annotations = coco.loadAnns(annotation_ids)

        # Dessiner les masques sur l'image
        for ann in annotations:
            mask = coco.annToMask(ann)
            image[mask == 1] = [255, 0, 0]  # Colorer les masques en rouge

        # Afficher l'image avec les annotations
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Une erreur est survenue lors du chargement de l'image : {e}")