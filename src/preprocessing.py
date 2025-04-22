import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import tensorflow as tf

def load_data(config):
    data_dir = config['data']['base_dir']
    labels_df = pd.read_csv(os.path.join(data_dir, config['data']['labels_file']))
    labels_df['id'] = labels_df['id'] + '.tif'
    labels_df['label'] = labels_df['label'].astype(str)
    return labels_df

def split_data(labels_df, val_split=0.2):
    train_df, val_df = train_test_split(labels_df, test_size=val_split, stratify=labels_df['label'], random_state=42)
    return train_df, val_df

def preprocess_image(image, config):
    image = cv2.resize(image, tuple(config['data']['image_size']))
    if config['preprocessing']['contrast_alpha']:
        image = cv2.convertScaleAbs(image, alpha=config['preprocessing']['contrast_alpha'], 
                                beta=config['preprocessing']['contrast_beta'])
    if config['preprocessing']['normalization']:
        image = image / 255.0
    return image

def get_data_generator(config, labels_df, train=True):
    aug_params = config['preprocessing']['augmentation'] if train else {}
    datagen = ImageDataGenerator(
        preprocessing_function=lambda x: preprocess_image(x, config),
        **aug_params
    )
    generator = datagen.flow_from_dataframe(
        dataframe=labels_df,
        directory=os.path.join(config['data']['base_dir'], config['data']['train_dir']),
        x_col='id',
        y_col='label',
        target_size=tuple(config['data']['image_size']),
        batch_size=config['data']['batch_size'],
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    # Wrap the generator in a tf.data.Dataset and repeat indefinitely
    dataset = tf.data.Dataset.from_generator(
        lambda: generator,
        output_types=(tf.float32, tf.float32),
        output_shapes=([None, *config['data']['image_size'], 3], [None])
    ).repeat()  # Repeat indefinitely
    return dataset, generator