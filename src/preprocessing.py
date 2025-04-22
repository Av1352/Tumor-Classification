# src/preprocessing.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(train_labels_path):
    """Load training labels from CSV."""
    train_labels = pd.read_csv(train_labels_path)
    print(f"Training labels shape: {train_labels.shape}")
    print(train_labels.head())
    return train_labels

def check_class_distribution(train_labels):
    """Print class distribution of labels."""
    class_distribution = train_labels['label'].value_counts(normalize=True) * 100
    print(f"\nClass distribution (%):\n{class_distribution}")

def create_data_generators(train_labels, train_dir, img_size, batch_size, validation_split):
    """Create train and validation data generators."""
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    train_labels['filename'] = train_labels['id'] + '.tif'
    train_labels['label_str'] = train_labels['label'].astype(str)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=train_dir,
        x_col='filename',
        y_col='label_str',
        subset='training',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train_labels,
        directory=train_dir,
        x_col='filename',
        y_col='label_str',
        subset='validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_generator, validation_generator

def create_test_generator(test_dir, img_size, batch_size):
    """Create test data generator for submission."""
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_files = os.listdir(test_dir)
    test_df = pd.DataFrame({
        'id': [os.path.splitext(file)[0] for file in test_files],
        'filename': test_files
    })

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col='filename',
        y_col=None,
        target_size=img_size,
        batch_size=batch_size * 2,
        class_mode=None,
        shuffle=False
    )
    return test_generator, test_df