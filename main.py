import os
import yaml
import tensorflow as tf
import numpy as np
from preprocessing import load_data, check_class_distribution, create_data_generators, create_test_generator
from model import build_baseline_model, build_batchnorm_model, compile_model
from evaluation import evaluate_model
from data_mining import perform_eda
from explainability import apply_gradcam, apply_shap

def main():
    # Load configuration
    with open('config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Define paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    models_dir = os.path.join(base_dir, 'models')
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Set random seed
    np.random.seed(config['model']['random_seed'])
    tf.random.set_seed(config['model']['random_seed'])

    # Check for GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Num GPUs Available: {len(gpus)}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using MirroredStrategy with {strategy.num_replicas_in_sync} T4 GPUs")
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("No GPUs available, using Kaggle standard CPU")
        tf.keras.mixed_precision.set_global_policy('float32')

    # Load and preprocess data
    train_labels = load_data(config['data']['train_labels'])
    check_class_distribution(train_labels)
    perform_eda(train_labels, config['data']['train_dir'])

    train_generator, validation_generator = create_data_generators(
        train_labels,
        config['data']['train_dir'],
        config['model']['img_size'],
        config['model']['batch_size'],
        config['training']['validation_split']
    )

    # Build and train models
    with strategy.scope():
        baseline_model = build_baseline_model(config['model']['img_size'])
        baseline_model = compile_model(baseline_model, config['training']['learning_rate'])
        
        batchnorm_model = build_batchnorm_model(config['model']['img_size'])
        batchnorm_model = compile_model(batchnorm_model, config['training']['learning_rate'])

    # Define callbacks with explicit monitoring and verbosity
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            verbose=1
        )
    ]

    print(f"Training baseline model for {config['model']['epochs']} epochs...")
    baseline_history = baseline_model.fit(
        train_generator,
        epochs=config['model']['epochs'],
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Baseline model training completed. Epochs run: {len(baseline_history.history['loss'])}")

    print(f"Training batchnorm model for {config['model']['epochs']} epochs...")
    batchnorm_history = batchnorm_model.fit(
        train_generator,
        epochs=config['model']['epochs'],
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    print(f"Batchnorm model training completed. Epochs run: {len(batchnorm_history.history['loss'])}")

    # Save models
    baseline_model_path = os.path.join(models_dir, 'baseline_model.keras')
    batchnorm_model_path = os.path.join(models_dir, 'batchnorm_model.keras')
    baseline_model.save(baseline_model_path)
    batchnorm_model.save(batchnorm_model_path)
    print(f"Baseline model saved to {baseline_model_path}")
    print(f"BatchNorm model saved to {batchnorm_model_path}")

    # Evaluate models
    baseline_metrics = evaluate_model(baseline_model, validation_generator, config)
    batchnorm_metrics = evaluate_model(batchnorm_model, validation_generator, config)
    print("Baseline Model Evaluation Metrics:", baseline_metrics)
    print("BatchNorm Model Evaluation Metrics:", batchnorm_metrics)

    # Select best model for submission
    baseline_accuracy = baseline_metrics.get('accuracy', 0)
    batchnorm_accuracy = batchnorm_metrics.get('accuracy', 0)
    if baseline_accuracy > batchnorm_accuracy:
        best_model = baseline_model
        best_model_path = baseline_model_path
        print(f"Using Baseline model for submission (accuracy: {baseline_accuracy})")
    else:
        best_model = batchnorm_model
        best_model_path = batchnorm_model_path
        print(f"Using BatchNorm model for submission (accuracy: {batchnorm_accuracy})")

    # Get a sample image and its label for visualizations
    sample_image, sample_label = next(train_generator)
    sample_image = sample_image[0]  # First image in batch
    sample_label = int(sample_label[0])  # First label in batch

    # Run Grad-CAM
    apply_gradcam(best_model, sample_image, config, sample_label=sample_label,
                  save_path=os.path.join(results_dir, 'gradcam.png'))
    print("Grad-CAM visualization saved to results/gradcam.png")

    # Run SHAP
    background = next(train_generator)[0]  # Background images for SHAP
    apply_shap(best_model, sample_image, background, config,
               save_path=os.path.join(results_dir, 'shap.png'))
    print("SHAP visualization saved to results/shap.png")

    # Generate submission
    test_generator, test_df = create_test_generator(
        config['data']['test_dir'],
        config['model']['img_size'],
        config['model']['batch_size']
    )

    print("\nGenerating predictions for submission...")
    predictions = best_model.predict(test_generator, verbose=1)
    predicted_classes = (predictions > 0.5).astype(int).flatten()

    submission_df = pd.DataFrame({
        'id': test_df['id'][:len(predicted_classes)],
        'label': predicted_classes
    })

    submission_path = os.path.join(results_dir, 'submission.csv')
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Sample of submission file:\n{submission_df.head()}")

if __name__ == "__main__":
    main()
