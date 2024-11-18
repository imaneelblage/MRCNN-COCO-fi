import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

class MaskRCNNConfig:
    IMAGE_SIZE = (640, 480)
    VAL_ANNOTATIONS = 'C:/Users/oassa/Downloads/PROJET IMANE/annotations/instances_val2017.json'
    VAL_IMAGES = 'C:/Users/oassa/Downloads/PROJET IMANE/val2017'

class ResizeLayer(tf.keras.layers.Layer):
    def __init__(self, target_size=(640, 480), **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({
            'target_size': self.target_size
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Le reste du code reste inchangé
class MaskRCNNDataset:
    def __init__(self, annotation_path, images_dir):
        self.annotation_path = annotation_path
        self.images_dir = images_dir

    def load_data(self, max_images=None):
        """Charge les données d'images et de masques"""
        image_paths = tf.io.gfile.glob(f"{self.images_dir}/*.jpg")
        if max_images:
            image_paths = image_paths[:max_images]

        images = []
        masks = []
        
        for image_path in image_paths:
            image = load_and_preprocess_image(image_path)
            if image is not None:
                images.append(image)
                mask_path = image_path.replace('images', 'annotations').replace('.jpg', '_mask.png')
                mask = load_and_preprocess_mask(mask_path)
                if mask is not None:
                    masks.append(mask)

        return np.array(images), np.array(masks)

def load_and_preprocess_image(image_path):
    try:
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, MaskRCNNConfig.IMAGE_SIZE)
        image = image / 255.0
        return image
    except Exception as e:
        print(f"Erreur lors du chargement de l'image {image_path}: {e}")
        return None

def load_and_preprocess_mask(mask_path):
    try:
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=1)
        mask = tf.image.resize(mask, MaskRCNNConfig.IMAGE_SIZE, method='nearest')
        mask = tf.cast(mask, tf.float32)
        background = tf.cast(tf.equal(mask, 0), tf.float32)
        foreground = tf.cast(tf.greater(mask, 0), tf.float32)
        mask_onehot = tf.stack([background, foreground], axis=-1)
        return mask_onehot
    except Exception as e:
        print(f"Erreur lors du chargement du masque {mask_path}: {e}")
        return None

def visualize_prediction(image, mask, prediction, save_path=None, display=True):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Image Originale')
    axes[0].axis('off')
    
    axes[1].imshow(mask[..., 1], cmap='gray')
    axes[1].set_title('Masque Réel')
    axes[1].axis('off')
    
    axes[2].imshow(prediction[..., 1], cmap='gray')
    axes[2].set_title('Prédiction')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    elif display:
        plt.show()

def evaluate_model(model_path='horse_segmentation_best.keras'):
    if not os.path.exists(model_path):
        print(f"Erreur : Le modèle {model_path} n'existe pas.")
        return

    custom_objects = {
        'ResizeLayer': ResizeLayer
    }
    
    try:
        model = load_model(model_path, custom_objects=custom_objects)
        print(f"Modèle chargé depuis {model_path}")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return
    
    val_dataset = MaskRCNNDataset(
        annotation_path=MaskRCNNConfig.VAL_ANNOTATIONS,
        images_dir=MaskRCNNConfig.VAL_IMAGES
    )
    X_val, y_val = val_dataset.load_data(max_images=10)
    
    predictions = model.predict(X_val)
    
    iou_scores = []
    dice_scores = []
    
    for i in range(len(X_val)):
        true_mask = y_val[i, ..., 1]
        pred_mask = predictions[i, ..., 1] > 0.5
        
        intersection = np.logical_and(true_mask, pred_mask).sum()
        union = np.logical_or(true_mask, pred_mask).sum()
        iou = intersection / (union + 1e-7)
        
        dice = (2 * intersection) / (true_mask.sum() + pred_mask.sum() + 1e-7)
        
        iou_scores.append(iou)
        dice_scores.append(dice)
        
        visualize_prediction(
            X_val[i],
            y_val[i],
            predictions[i],
            save_path=f'prediction_results_{i}.png',
            display=False
        )
    
    print(f"IoU moyen: {np.mean(iou_scores):.4f}")
    print(f"Dice Score moyen: {np.mean(dice_scores):.4f}")

if __name__ == "__main__":
    model_path = 'horse_segmentation_best.keras'
    evaluate_model(model_path)