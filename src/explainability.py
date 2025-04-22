import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
import shap
import warnings

# Suppress TensorFlow/SHAP warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")

def apply_gradcam(model, image, config, sample_label=None, save_path=None):
    """
    Apply Grad-CAM to visualize important regions in the image for model predictions.
    
    Args:
        model: Trained Keras model.
        image: Input image (numpy array, shape: (height, width, channels)).
        config: Configuration dictionary with explainability settings.
        sample_label: Ground truth label for the image (optional, for reference).
        save_path: Path to save the visualization (optional).
    """
    gradcam = Gradcam(model, model_modifier=lambda m: None, clone=False)
    
    def score_function(output):
        return output[:, 0]  # Probability of positive class (cancer)
    
    # Generate heatmap
    heatmap = gradcam(
        score_function,
        np.expand_dims(image, axis=0),
        penultimate_layer=config['explainability']['gradcam_layer'],
        seek_penultimate_conv_layer=False
    )
    
    # Normalize heatmap to [0, 1] for better visualization
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Create a figure with two subplots: original image and heatmap overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Plot original image
    ax1.imshow(image)
    ax1.set_title(f'Original Image (Label: {sample_label if sample_label is not None else "Unknown"})')
    ax1.axis('off')
    
    # Plot heatmap overlay
    ax2.imshow(image)
    ax2.imshow(heatmap[0], cmap='jet', alpha=0.5)
    ax2.set_title('Grad-CAM Heatmap')
    ax2.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Grad-CAM visualization saved to {save_path}")
    plt.close()

def apply_shap(model, image, background, config, save_path=None):
    """
    Apply SHAP to explain model predictions by attributing contributions to image regions.
    
    Args:
        model: Trained Keras model.
        image: Input image (numpy array, shape: (height, width, channels)).
        background: Background images for SHAP (numpy array, shape: (n_samples, height, width, channels)).
        config: Configuration dictionary with explainability settings.
        save_path: Path to save the visualization (optional).
    """
    explainer = shap.DeepExplainer(model, background[:config['explainability']['shap_background_samples']])
    shap_values = explainer.shap_values(np.expand_dims(image, axis=0))
    
    # Normalize SHAP values to [0, 1]
    shap_values_normalized = [(sv - sv.min()) / (sv.max() - sv.min()) for sv in shap_values]
    shap.image_plot(shap_values_normalized, np.expand_dims(image, axis=0), show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"SHAP visualization saved to {save_path}")
    plt.close()

