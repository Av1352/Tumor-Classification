import pandas as pd
import matplotlib.pyplot as plt

def analyze_label_distribution(labels_df, save_path=None):
    label_counts = labels_df['label'].value_counts()
    print("Label Distribution:\n", label_counts)
    
    plt.bar(['Non-Cancerous', 'Cancerous'], label_counts)
    plt.title('Label Distribution')
    if save_path:
        plt.savefig(save_path)
    plt.close()