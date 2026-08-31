# **ASL Fingerspelling Letter Classifier**

A CNN trained from scratch to recognize American Sign Language (ASL) letters from a single static hand-shape image. This is the foundational building block a full fingerspelling reader would sit on top of — one that reads a sequence of hand shapes and turns them into text for real-time captioning.

Problem

Given a photo of a hand shape, classify it into the correct ASL letter.

The task is scoped to a single still image on purpose: J and Z both require hand motion to sign and can't be represented in one static frame, so the problem reduces to a 24-class classification task (A–Z minus J and Z).

Dataset

Sign Language MNIST — a published Kaggle dataset of ASL hand-sign images.

27,455 training images, 7,172 test images
Each image: 28x28 grayscale, flattened into a single CSV row (784 pixel columns)
24 classes (A–Z, excluding J and Z)
Built by cropping a small set of original hand photos to the hand, then augmenting them (rotation, brightness/contrast, filtering) to generate the full dataset
Approach
EDA — reshaped sample rows back into images, checked class balance across the 24 letters, and specifically visualized letters expected to be visually similar (M/N/S, R/U/V/K) to set an expectation for where the model would likely struggle.
Preprocessing — normalized pixel values to [0, 1], remapped labels to a contiguous 0–23 range (raw labels have gaps at 9/J and 25/Z, which breaks a 24-unit softmax otherwise), and split off a stratified validation set from the training data. The official test set was left untouched until final evaluation.
Baseline sanity check — a minimal CNN with no regularization or augmentation, trained briefly to confirm the task was actually learnable from this data before investing in a more complex architecture.
Final architecture — a CNN designed from scratch (not a pretrained backbone), with choices driven by the specific problem rather than picked arbitrarily (see below).
Evaluation — accuracy alone isn't enough for a 24-class problem, so the notebook also reports macro F1, balanced accuracy, per-class precision/recall, a full confusion matrix, and a visual gallery of the actual misclassified images with the model's confidence for each.
Architecture
Two convolutional layers per stage at the first two resolutions (32, then 64 filters), giving the network more capacity to separate similar hand shapes while spatial resolution is still high.
Only two pooling stages (28→14→7), not three. An earlier version pooled a third time down to 3x3 and plateaued around 94% test accuracy — that level of downsampling destroys the fine finger-position detail that separates letters like M, N, and S. The final conv block stays at 7x7 instead.
Conv → BatchNorm → Activation ordering, with use_bias=False on the conv layers (BatchNorm's own shift makes the bias redundant) — trains more stably than normalizing after the activation.
Dropout increasing with depth (0.25 → 0.25 → 0.30 → 0.5 on the dense head), targeting where overfitting hits hardest given how close to duplicate the augmented training images are within a class.
Light augmentation only — small rotation, zoom, and translation. The dataset is already built from augmented copies of a small photo set, so heavy augmentation on top risks distorting hand shapes past recognition.
Results
Metric	Score
Test accuracy	99.97%
Misclassified	2 / 7,172
Macro F1	0.9997
Balanced accuracy	0.9997
A real debugging story

An earlier version of this model got stuck at random-chance accuracy (~4%) for over 20 epochs before breaking loose — loss frozen near ln(24) ≈ 3.18, the signature of a network outputting near-uniform guesses. Diagnosing this (rather than guessing at fixes) involved:

A minimal baseline model to isolate whether the problem was in the data or the architecture
Fixing a tf.data pipeline ordering bug that caused stale variables to leak into training
Reworking the architecture (pooling depth, conv/BatchNorm/activation ordering, regularization schedule) based on a specific hypothesis about what information the network needed to preserve
Caveat on the reported accuracy

Near-100% accuracy on a 24-class hand-shape problem is unusually high, and is a known characteristic of this dataset — both train and test images are augmented copies of a small set of original source photos, so some shared visual origin between splits is plausible. Accuracy on genuinely novel, independently-taken hand photos would very likely be meaningfully lower than 99.97%. Testing on real, independently-taken photos is a natural next step.

Repo contents
fingerspelling-letter-classifier.ipynb — full notebook: EDA, preprocessing, training, evaluation
asl_letter_cnn.keras — trained model weights
requirements.txt — Python dependencies
Setup
bash
pip install -r requirements.txt

Download the dataset from Kaggle and place sign_mnist_train.csv / sign_mnist_test.csv in the path referenced at the top of the notebook, or update the path to match your environment.

What's next

This notebook solves single-letter recognition from a static image. The next stage toward a full real-time captioning tool is a sequence model that reads a series of hand-shape predictions and assembles them into fingerspelled words.
