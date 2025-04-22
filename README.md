# TumorClassificationXAI

This project tackles the Histopathologic Cancer Detection challenge, aiming to identify metastatic cancer in histopathology images. The dataset consists of 96x96 pixel images, with labels based on the presence of tumor tissue in the center 32x32 pixel region. The project compares two CNN architectures: a baseline model and one with batch normalization.

## Project Structure
- `data/`: Contains train, test, and train_labels.csv.
- `src/`: Python modules for preprocessing, model definition, evaluation, and execution.
- `config/`: Configuration file for hyperparameters and paths.
- `requirements.txt`: Lists dependencies.
- `README.md`: Project documentation.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the dataset is placed in the `data/` directory.
3. Run the main script: `python src/main.py`

## Results
- Baseline CNN: Validation accuracy ~0.85, AUC ~0.93
- BatchNorm CNN: Validation accuracy ~0.91, AUC ~0.97

## Future Work
- Implement data augmentation preserving the center 32x32 region.
- Explore different activation functions.
- Develop explainable AI methods to visualize model attention.