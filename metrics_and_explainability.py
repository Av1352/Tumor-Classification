import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus
import shap
import warnings
import tensorflow as tf

# Suppress TensorFlow/SHAP/Keras warnings
warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.models.functional")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

def load_config(config_path='config/config.yaml'):
    """Load configuration from YAML file with default values."""
    default_config = {
        'data': {
            'train_dir': 'data/train/',
            'train_labels': 'data/train_labels.csv',
            'image_size': [96, 96],
            'batch_size': 128
        },
        'training': {
            'validation_split': 0.2,
            'learning_rate': 0.001
        },
        'results_dir': 'results',
        'explainability': {
            'gradcam_layer': 'conv2d_5',  # Last Conv2D layer from print_model_layers
            'shap_background_samples': 50
        }
    }
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        def merge_dicts(default, loaded):
            for key, value in loaded.items():
                if isinstance(value, dict) and key in default:
                    default[key] = merge_dicts(default[key], value)
                else:
                    default[key] = value
            return default
        config = merge_dicts(default_config, config)
    except FileNotFoundError:
        print(f"Config file {config_path} not found. Using default configuration.")
        config = default_config
    
    return config

def load_data(labels_path):
    """Load training labels and prepare for generator."""
    labels_df = pd.read_csv(labels_path)
    labels_df['id'] = labels_df['id'] + '.tif'
    labels_df['label'] = labels_df['label'].astype(str)
    return labels_df

def create_validation_generator(labels_df, train_dir, img_size, batch_size, validation_split):
    """Create validation data generator."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    val_df = labels_df.sample(frac=validation_split, random_state=42)
    generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        subset='validation',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )
    return generator

def create_train_generator(labels_df, train_dir, img_size, batch_size, validation_split):
    """Create training data generator for explainability sampling."""
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )
    train_df = labels_df.sample(frac=1.0 - validation_split, random_state=42)
    generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col='id',
        y_col='label',
        subset='training',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True
    )
    return generator

def print_model_layers(model):
    """Print model layer names and output shapes for debugging."""
    print("\nModel Layer Names and Output Shapes:")
    for layer in model.layers:
        print(f"Layer: {layer.name}, Type: {layer.__class__.__name__}")
    print()

def custom_gradcam(model, image, layer_name):
    """Custom Grad-CAM implementation to bypass tf_keras_vis issues."""
    from tensorflow.keras.models import Model
    import tensorflow as tf
    
    try:
        # Perform a forward pass to initialize the model
        _ = model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        # Create a model that outputs the specified layer's activations and final predictions
        layer_output = model.get_layer(layer_name).output
        grad_model = Model(inputs=model.inputs, outputs=[layer_output, model.output])
        
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.expand_dims(image, axis=0))
            loss = predictions[:, 0]  # Positive class score
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError(f"Gradients are None for layer '{layer_name}'.")
        grads = grads[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    except Exception as e:
        print(f"Custom Grad-CAM failed for layer '{layer_name}': {str(e)}")
        return None

def simplified_gradcam(model, image, layer_name):
    """Simplified Grad-CAM using direct GradientTape on Sequential model."""
    import tensorflow as tf
    
    try:
        # Perform a forward pass to initialize the model
        _ = model.predict(np.expand_dims(image, axis=0), verbose=0)
        
        with tf.GradientTape() as tape:
            # Get the layer output
            layer_output = model.get_layer(layer_name).output
            temp_model = tf.keras.Model(inputs=model.inputs, outputs=[layer_output, model.output])
            conv_outputs, predictions = temp_model(np.expand_dims(image, axis=0))
            loss = predictions[:, 0]  # Positive class score
        
        # Compute gradients
        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError(f"Gradients are None for layer '{layer_name}'.")
        grads = grads[0]
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature maps
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(conv_outputs, pooled_grads), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        
        return heatmap.numpy()
    except Exception as e:
        print(f"Simplified Grad-CAM failed for layer '{layer_name}': {str(e)}")
        return None

def apply_gradcam(model, image, config, sample_label=None, save_path=None):
    """Apply Grad-CAM to visualize important regions in the image."""
    try:
        print(f"Input image shape: {image.shape}")
        prediction = model.predict(np.expand_dims(image, axis=0))
        print(f"Model prediction: {prediction}")
        
        gradcam_layer = config['explainability']['gradcam_layer']
        conv_layers = [layer.name for layer in model.layers if layer.__class__.__name__ == 'Conv2D']
        print(f"Available Conv2D layers: {conv_layers}")
        
        try:
            layer = model.get_layer(gradcam_layer)
            print(f"Selected layer: {gradcam_layer}, Type: {layer.__class__.__name__}")
            if not layer.__class__.__name__.startswith('Conv'):
                print(f"Warning: Layer '{gradcam_layer}' is not a Conv layer.")
                gradcam_layer = None
        except ValueError as e:
            if str(e).startswith("No such layer"):
                print(f"Error: Layer '{gradcam_layer}' not found.")
                gradcam_layer = None
        
        # Try Grad-CAM, Grad-CAM++, Custom Grad-CAM, and Simplified Grad-CAM
        for method, gradcam in [
            ("Grad-CAM", Gradcam(model, model_modifier=lambda m: None, clone=False)),
            ("Grad-CAM++", GradcamPlusPlus(model, model_modifier=lambda m: None, clone=False)),
            ("Custom Grad-CAM", None),
            ("Simplified Grad-CAM", None)
        ]:
            for layer_name in ([gradcam_layer] if gradcam_layer else []) + conv_layers[::-1]:
                if not layer_name:
                    continue
                print(f"Attempting {method} with layer: {layer_name}")
                try:
                    if method == "Custom Grad-CAM":
                        heatmap = custom_gradcam(model, image, layer_name)
                    elif method == "Simplified Grad-CAM":
                        heatmap = simplified_gradcam(model, image, layer_name)
                    else:
                        def score_function(output):
                            return output[:, 0]  # Positive class score
                        heatmap = gradcam(
                            score_function,
                            np.expand_dims(image, axis=0),
                            penultimate_layer=layer_name,
                            seek_penultimate_conv_layer=False
                        )
                    if heatmap is None:
                        print(f"{method} returned None for layer '{layer_name}'.")
                        continue
                    print(f"{method} succeeded for layer: {layer_name}")
                    break
                except Exception as e:
                    print(f"{method} failed for layer '{layer_name}': {str(e)}")
                    continue
            else:
                continue
            break
        else:
            raise ValueError("All Grad-CAM methods failed for all Conv2D layers.")
        
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(image)
        ax1.set_title(f'Original Image (Label: {sample_label if sample_label is not None else "Unknown"})')
        ax1.axis('off')
        ax2.imshow(image)
        ax2.imshow(heatmap, cmap='jet', alpha=0.5)
        ax2.set_title('Grad-CAM Heatmap')
        ax2.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Grad-CAM failed: {str(e)}")
        if save_path:
            print(f"No Grad-CAM visualization saved to {save_path}")

def apply_shap(model, image, background, config, save_path=None):
    """Apply SHAP to explain model predictions by attributing contributions."""
    try:
        explainer = shap.DeepExplainer(model, background[:config['explainability']['shap_background_samples']])
        shap_values = explainer.shap_values(np.expand_dims(image, axis=0))
        shap_values_normalized = [(sv - sv.min()) / (sv.max() - sv.min()) for sv in shap_values]
        shap.image_plot(shap_values_normalized, np.expand_dims(image, axis=0), show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"SHAP visualization saved to {save_path}")
        plt.close()
    except Exception as e:
        print(f"SHAP failed: {str(e)}")
        if save_path:
            print(f"No SHAP visualization saved to {save_path}")

def evaluate_model(model, generator, results_dir):
    """Evaluate model and compute metrics: accuracy, AUC, confusion matrix, ROC curve."""
    os.makedirs(results_dir, exist_ok=True)
    generator.reset()
    predictions = model.predict(generator, verbose=1)
    true_labels = generator.classes
    predicted_classes = (predictions > 0.5).astype(int).flatten()
    accuracy = np.mean(predicted_classes == true_labels)
    print(f"Accuracy: {accuracy:.4f}")
    cm = confusion_matrix(true_labels, predicted_classes)
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_classes))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    cm_save_path = os.path.join(results_dir, 'confusion_matrix.png')
    plt.savefig(cm_save_path, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {cm_save_path}")
    fpr, tpr, _ = roc_curve(true_labels, predictions)
    roc_auc = auc(fpr, tpr)
    print(f"AUC: {roc_auc:.4f}")
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    roc_save_path = os.path.join(results_dir, 'roc_curve.png')
    plt.savefig(roc_save_path, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to {roc_save_path}")
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'confusion_matrix': cm,
        'fpr': fpr,
        'tpr': tpr
    }

def apply_explainability(model, train_generator, config, results_dir):
    """Apply Grad-CAM and SHAP visualizations to a sample image."""
    os.makedirs(results_dir, exist_ok=True)
    train_generator.reset()
    sample_image, sample_label = next(train_generator)
    sample_image = sample_image[0]
    sample_label = int(sample_label[0])
    gradcam_save_path = os.path.join(results_dir, 'gradcam.png')
    apply_gradcam(model, sample_image, config, sample_label=sample_label, save_path=gradcam_save_path)
    background = next(train_generator)[0]
    shap_save_path = os.path.join(results_dir, 'shap.png')
    apply_shap(model, sample_image, background, config, save_path=shap_save_path)

def main():
    tf.keras.backend.clear_session()  # Clear session to avoid naming conflicts
    config = load_config()
    base_dir = os.path.abspath(os.path.dirname(__file__))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, config.get('results_dir', 'results'))
    labels_df = load_data(config['data']['train_labels'])
    validation_generator = create_validation_generator(
        labels_df,
        config['data']['train_dir'],
        config['data']['image_size'],
        config['data']['batch_size'],
        config['training']['validation_split']
    )
    train_generator = create_train_generator(
        labels_df,
        config['data']['train_dir'],
        config['data']['image_size'],
        config['data']['batch_size'],
        config['training']['validation_split']
    )
    model_path = os.path.join(models_dir, 'baseline_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = load_model(model_path, compile=False)
    model.compile(optimizer=Adam(learning_rate=config['training']['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    print(f"Model loaded from {model_path}")
    print_model_layers(model)  # Print layer names for debugging
    model.summary()  # Print model summary for additional context
    metrics = evaluate_model(model, validation_generator, results_dir)
    print("\nEvaluation Metrics:", metrics)
    apply_explainability(model, train_generator, config, results_dir)

if __name__ == "__main__":
    main()