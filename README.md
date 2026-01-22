
# Neural Network Chess Engine

This project implements a Convolutional Neural Network (CNN) trained on high-level chess games to predict moves. It features a "Smart Move" wrapper that combines the neural network's intuition with a tactical blunder check to ensure solid play.

## Overview

The engine treats chess as a computer vision problem. It takes the board state as an 8x8x12 image (representing piece locations) and outputs a probability distribution over 1,883 possible moves.

Key features:

* **CNN Architecture:** Uses a custom VGG-style architecture with a "bottleneck" layer to reduce parameters and prevent overfitting.
* **Blunder Checking:** A wrapper function that filters the network's top 5 predictions to prevent hanging pieces (1-ply lookahead).
* **GPU Optimization:** Implements TensorFlow `MirroredStrategy` and `Mixed Precision` to maximize training speed on single or dual GPUs.
* **Data Pipeline:** Automatically downloads, processes, and streams data from the Lichess Elite Database.

## Prerequisites

To run the notebook or the standalone script, you need the following Python libraries:

```bash
pip install tensorflow numpy chess

```

*Note: For GPU acceleration, ensure you have the appropriate CUDA and cuDNN drivers installed for your hardware.*

## Project Structure

* **Training Notebook:** Contains the complete pipeline for downloading data, preprocessing, model definition, and training.
* **chess_engine.keras:** The trained model file (generated after training).
* **move_map.json:** The vocabulary file mapping model output indices to UCI move strings (e.g., "e2e4").

## Usage

### 1. Training the Model

Open the notebook in Jupyter or Kaggle. Run the cells in order to:

1. Download the dataset (Lichess Elite games).
2. Process the PGN files into matrix inputs.
3. Train the CNN using the optimized `tf.data` pipeline.
4. Save the model (`chess_engine.keras`) and vocabulary (`move_map.json`).

### 2. Playing Against the AI

Once the model and map files are generated, you can play against the engine using the provided testing script.

1. Ensure `chess_engine.keras` and `move_map.json` are in the same directory.
2. Run the testing script:

```python
python run_chess.py

```

3. Enter moves using UCI format (e.g., `e2e4`, `g1f3`).

## Model Architecture

The model input is an **8x8x12** tensor:

* 8x8 Grid: The chess board.
* 12 Channels: 6 piece types for White + 6 piece types for Black.

**Layers:**

1. **Conv2D Block 1:** 64 filters (3x3), Batch Normalization, ReLU.
2. **Conv2D Block 2:** 128 filters (3x3), Batch Normalization, ReLU.
3. **Conv2D Block 3:** 256 filters (3x3), Batch Normalization, ReLU.
4. **Bottleneck Layer:** 64 filters (1x1) to compress depth and reduce parameters.
5. **Flatten & Dense:** A 512-neuron dense layer for global reasoning.
6. **Output:** Softmax layer over the vocabulary size (approx. 1,883 unique moves).

## Performance

On a dataset of 25,000 games (approx. 2 million positions), the model typically achieves:

* **Top-1 Accuracy:** ~32% (Exact match with Grandmaster move).
* **Top-5 Accuracy:** ~65% (Grandmaster move is in the top 5 suggestions).

## Credits

* **Dataset:** Lichess Elite Database (nikonoel.fr).
* **Libraries:** TensorFlow, python-chess.
