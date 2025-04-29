![type_win_rates](https://github.com/user-attachments/assets/7801f316-c732-4ba6-a366-b2d055b1ce4e)
# Pokemon Battle Prediction Project

This project uses machine learning to predict the outcome of Pokemon battles based on Pokemon stats and types. It includes data preprocessing, model training, evaluation, and prediction components. https://www.kaggle.com/datasets/terminus7/pokemon-challenge

## Overview

The system uses neural networks to learn patterns from historical battle data, considering Pokemon attributes such as HP, Attack, Defense, Speed, and Types to predict which Pokemon will win in a battle.

## Project Structure

- **Data Exploration**: Analyze Pokemon stats and battle outcomes
- **Data Preprocessing**: Clean and transform raw data into model-ready features
- **Model Training**: Train neural networks with various architectures
- **Prediction**: Generate battle outcome predictions using trained models
- **Evaluation**: Compare model performance and analyze results

## Requirements

- numpy>=1.26.0,<2.0.0
- pandas==2.0.3
- matplotlib==3.7.3
- scikit-learn==1.3.2
- tensorflow>=2.15.0,<2.16.0
- Keras (installed automatically as a dependency of TensorFlow)

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Data Exploration

Run the data exploration script to analyze the Pokemon dataset and battle records:

```
python data_exploration.py
```

This script generates visualizations in the `plots` directory.

### Training a Model

Train a neural network model with customizable parameters:

```
python pokemon_neural_net_train.py --layers 128 64 32 --activation relu --dropout 0.2 --optimizer adam --learning-rate 0.001 --epochs 100 --exp-name my_pokemon_model
```

Key parameters:
- `--layers`: Hidden layer sizes (space-separated)
- `--activation`: Activation function (relu, tanh, etc.)
- `--dropout`: Dropout rate for regularization
- `--optimizer`: Training optimizer (adam, sgd, rmsprop, adagrad)
- `--learning-rate`: Initial learning rate
- `--epochs`: Maximum training epochs
- `--exp-name`: Optional experiment name (auto-generated if not provided)

### Making Predictions

Generate predictions using a trained model:

```
python pokemon_neural_net_predict.py --model my_pokemon_model_model.keras --test-features pokemon_X_test.csv
```

### Comparing Models

Compare the predictions of two different models:

```
python compare_predictions.py --predictions1 model1_predictions.csv --predictions2 model2_predictions.csv
```

### Generating Pokemon Rankings

Generate rankings based on battle performance:

```
python pokemon_rankings.py --min-battles 15 --output-dir rankings
```

## Model Architecture

The neural network architecture is configurable with:
- Variable number and size of hidden layers
- Different activation functions
- Dropout regularization
- Various optimizers and learning rates

## Results

Models are evaluated using:
- Accuracy
- AUC (Area Under ROC Curve)
- Precision
- Recall

Learning curves and model metrics are saved for each experiment in the format:
- `{experiment_name}_model.keras`: Trained model
- `{experiment_name}_learning_curve.png`: Training/validation metrics plot
- `experiments/{experiment_name}_results.json`: Detailed results

## Pokemon Rankings

The system can generate Pokemon battle performance rankings based on historical win rates, with visualizations showing:
- Top Pokemon by win rate
- Type effectiveness analysis
- Win rate distributions
