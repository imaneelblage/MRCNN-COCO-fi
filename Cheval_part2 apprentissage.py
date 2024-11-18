import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications import ResNet50
from pycocotools.coco import COCO



print(f"TensorFlow version: {tf.__version__}")

class horseConfig:
    """Configuration pour le modèle de segmentation d'chevaux"""
    NAME = "horse_segmentation"
    IMAGES_PER_GPU = 2
    BATCH_SIZE = IMAGES_PER_GPU
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 2  # Background + horse
    IMAGE_SIZE = (640, 480)
    BACKBONE = 'resnet50'
    
    # Chemins des données
    TRAIN_ANNOTATIONS = 'C:/Users/oassa/Downloads/PROJET IMANE/annotations/instances_train2017.json'
    TRAIN_IMAGES = 'C:/Users/oassa/Downloads/PROJET IMANE/train2017/train2017'
    VAL_ANNOTATIONS = 'C:/Users/oassa/Downloads/PROJET IMANE/annotations/instances_val2017.json'
    VAL_IMAGES = 'C:/Users/oassa/Downloads/PROJET IMANE/val2017'

class ResizeLayer(layers.Layer):
    """Couche personnalisée pour le redimensionnement des images"""
    def __init__(self, target_size, **kwargs):
        super(ResizeLayer, self).__init__(**kwargs)
        self.target_size = target_size

    def call(self, inputs):
        return tf.image.resize(inputs, self.target_size)

    def get_config(self):
        config = super(ResizeLayer, self).get_config()
        config.update({"target_size": self.target_size})
        return config

class horseDataset:
    """Classe pour gérer le dataset d'chevaux"""
    def __init__(self, annotation_path, images_dir):
        """Initialise le dataset"""
        self.images_dir = images_dir
        try:
            self.coco = COCO(annotation_path)
            print(f"Chargement réussi des annotations depuis {annotation_path}")
        except Exception as e:
            print(f"Erreur lors du chargement des annotations: {e}")
            raise
            
        self.horse_ids = self.coco.getCatIds(catNms=['horse'])
        if not self.horse_ids:
            print("Attention: Aucune catégorie 'horse' trouvée dans les annotations")
            print("Catégories disponibles:", self.coco.loadCats(self.coco.getCatIds()))
            
        self.img_ids = self.coco.getImgIds(catIds=self.horse_ids)
        print(f"Nombre d'images trouvées: {len(self.img_ids)}")

    def load_data(self, max_images=None):
        """Charge et prépare les données"""
        images = []
        masks = []
        valid_image_count = 0
        
        img_ids = self.img_ids[:max_images] if max_images else self.img_ids
        
        for img_id in img_ids:
            try:
                # Charger l'image
                img_info = self.coco.loadImgs([img_id])[0]
                image_path = os.path.join(self.images_dir, img_info["file_name"])
                
                if not os.path.exists(image_path):
                    print(f"Image non trouvée: {image_path}")
                    continue
                
                # Charger et prétraiter l'image
                image = tf.io.read_file(image_path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.cast(image, tf.float32)
                image = tf.image.resize(image, horseConfig.IMAGE_SIZE)
                
                # Charger les annotations
                anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=self.horse_ids, iscrowd=False)
                anns = self.coco.loadAnns(anns_ids)
                
                if not anns:
                    continue
                
                # Créer un masque binaire
                mask = np.zeros(horseConfig.IMAGE_SIZE + (2,), dtype=np.float32)
                
                for ann in anns:
                    # Convertir le masque COCO en masque binaire
                    m = self.coco.annToMask(ann)
                    m = tf.image.resize(
                        m[..., np.newaxis],
                        horseConfig.IMAGE_SIZE,
                        method='nearest'
                    ).numpy()
                    # Mettre à jour le canal du cheval (classe 1)
                    mask[..., 1] = np.maximum(mask[..., 1], m[..., 0])
                
                # Le fond (classe 0) est l'inverse du masque du cheval
                mask[..., 0] = 1 - mask[..., 1]
                
                images.append(image.numpy())
                masks.append(mask)
                valid_image_count += 1
                
                if valid_image_count % 100 == 0:
                    print(f"Chargées {valid_image_count} images valides")
                    
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {img_id}: {e}")
                continue
        
        if not images:
            raise ValueError("Aucune image valide n'a été chargée!")
            
        print(f"Chargement terminé: {len(images)} images valides traitées")
        return np.array(images), np.array(masks)

def create_mask_rcnn_model(config):
    """Crée un modèle de segmentation basé sur ResNet50"""
    # Backbone ResNet50
    backbone = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=config.IMAGE_SIZE + (3,)
    )
    
    # Feature Pyramid Network
    C5 = backbone.get_layer('conv5_block3_out').output
    P5 = layers.Conv2D(256, (1, 1), name='fpn_c5p5')(C5)
    P5_upsampled = layers.UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5)
    
    C4 = backbone.get_layer('conv4_block6_out').output
    P4 = layers.Conv2D(256, (1, 1), name='fpn_c4p4')(C4)
    P4 = layers.Add(name='fpn_p4add')([P5_upsampled, P4])
    
    # Mask head
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(P4)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    
    # Upsampling progressif
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(x)
    
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(16, (3, 3), padding="same", activation="relu")(x)
    
    # Utiliser ResizeLayer pour le redimensionnement final
    x = ResizeLayer(config.IMAGE_SIZE)(x)
    
    # Couche de sortie
    mask_output = layers.Conv2D(
        config.NUM_CLASSES, (1, 1),
        activation='softmax',
        name='mask_output'
    )(x)
    
    # Créer et compiler le modèle
    model = Model(inputs=backbone.input, outputs=mask_output)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Input shape: {config.IMAGE_SIZE + (3,)}")
    print(f"Output shape: {model.output_shape}")
    
    return model

def create_train_callbacks(model_name):
    """Crée les callbacks pour l'entraînement"""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            f'{model_name}_best.keras',
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=3,
            monitor='val_loss',
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f'./logs/{model_name}',
            update_freq='epoch'
        )
    ]

def main():
    """Fonction principale d'entraînement"""
    # Configuration
    config = horseConfig()
    
    try:
        # Charger et préparer les données d'entraînement
        print("Chargement du dataset d'entraînement...")
        train_dataset = horseDataset(
            annotation_path=config.TRAIN_ANNOTATIONS,
            images_dir=config.TRAIN_IMAGES
        )
        X_train, y_train = train_dataset.load_data(max_images=1000)
        
        # Charger et préparer les données de validation
        print("Chargement du dataset de validation...")
        val_dataset = horseDataset(
            annotation_path=config.VAL_ANNOTATIONS,
            images_dir=config.VAL_IMAGES
        )
        X_val, y_val = val_dataset.load_data(max_images=200)
        
        # Normaliser les images
        print("Normalisation des données...")
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        
        # Créer le modèle
        print("Création du modèle...")
        model = create_mask_rcnn_model(config)
        
        # Créer les callbacks
        callbacks = create_train_callbacks(config.NAME)
        
        # Entraîner le modèle
        print("Début de l'entraînement...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=config.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Sauvegarder le modèle final
        model.save(f'{config.NAME}_final.keras')
        print("Entraînement terminé et modèle sauvegardé!")
        
        # Sauvegarder l'historique d'entraînement
        np.save(f'{config.NAME}_history.npy', history.history)
        
    except Exception as e:
        print(f"Erreur dans le processus: {e}")
        raise

if __name__ == "__main__":
    main()
