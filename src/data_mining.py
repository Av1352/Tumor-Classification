# src/data_mining.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda(train_labels, train_dir):
    """Perform exploratory data analysis."""
    # Sample images from each class
    pos_samples = train_labels[train_labels['label'] == 1].sample(2)
    neg_samples = train_labels[train_labels['label'] == 0].sample(2)
    sample_df = pd.concat([pos_samples, neg_samples])

    sample_df['filename'] = sample_df['id'] + '.tif'
    sample_df['label_str'] = sample_df['label'].astype(str)

    # Plot images
    plt.figure(figsize=(15, 8))
    for i, (_, row) in enumerate(sample_df.iterrows()):
        img_path = os.path.join(train_dir, row['filename'])
        img = plt.imread(img_path) / 255
        plt.subplot(2, 4, i + 1)
        plt.imshow(img)
        plt.title(f"Label: {row['label']}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Plot distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='label', data=train_labels)
    plt.title('Training Labels Distribution')
    plt.xlabel('Label (0 = No Cancer, 1 = Cancer)')
    plt.ylabel('Count')
    plt.show()