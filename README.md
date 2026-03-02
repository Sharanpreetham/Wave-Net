Overview
This repository contains a PyTorch implementation of a character-level language model trained on a dataset of names (names.txt). Unlike standard Multi-Layer Perceptrons (MLPs) that flatten the entire context window at once, this model adopts a WaveNet-style hierarchical architecture. It progressively fuses pairs of adjacent character representations across multiple layers, allowing the network to build complex features over a context window of 8 characters.
All core neural network components—including Linear layers, Batch Normalization, and Embeddings—are implemented from scratch to provide a deep understanding of the underlying math and mechanics.

Key Features
Custom Neural Network API: Implements standard PyTorch-like modules from scratch, including Linear, BatchNorm1d, Tanh, Embedding, and a Sequential container.

Hierarchical Flattening (FlattenCons): Instead of squashing the entire context embedding at once, this custom module fuses consecutive elements progressively (e.g., 8 tokens -> 4 pairs -> 2 pairs -> 1) to simulate dilated causal convolutions.

Character-Level Tokenization: Builds a string-to-integer (sti) and integer-to-string (its) vocabulary dynamically from the dataset.

Mini-Batch Training: Features a custom training loop with a 200,000-step iteration, tracking cross-entropy loss, and utilizing learning rate decay for stabilization.

Inference/Sampling: Uses torch.multinomial to sample from the softmax probability distribution, generating highly realistic but entirely novel names character by character.

Model Architecture
The model processes an 8-character context window with the following pipeline:

Embedding Layer: Maps 27 vocabulary characters (26 letters + 1 padding/end token) into a 24-dimensional space.

Hidden Block 1: Flattens adjacent pairs (reducing sequence to 4) -> Linear Layer (128 neurons) -> BatchNorm -> Tanh activation.

Hidden Block 2: Flattens adjacent pairs (reducing sequence to 2) -> Linear Layer (128 neurons) -> BatchNorm -> Tanh activation.

Hidden Block 3: Flattens adjacent pairs (reducing sequence to 1) -> Linear Layer (128 neurons) -> BatchNorm -> Tanh activation.

Output Layer: Maps the final 128-dimensional hidden state to 27 logits representing the vocabulary size.

Total tunable parameters: ~76,579

Dataset
The model is trained on a provided names.txt file, split into three datasets to ensure robust evaluation:

Training Set: 80%

Validation Set: 10%

Test Set: 10%

Sample Output
After training, the model generates novel names that mimic the phonetic structure of the training data. Examples include:

fergi.

aliana.

graylynn.

giuliana.

elinorrose.
