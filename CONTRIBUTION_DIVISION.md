# Project Contribution Division Plan

This document outlines the proposed division of the Cloud Removal project into 4 distinct areas for team member contributions.

## Member 1: Model Architecture & Core Network
**Responsibility:** Design and Implementation of the Deep Learning Models.
**Key Components:**
*   **Cross Attention Network:** `codes/net_CR_CrossAttention.py`
    *   *Task:* Explain the main architecture using cross-attention mechanisms for cloud removal.
*   **Backbone (RDN):** `codes/net_CR_RDN.py`
    *   *Task:* Explain the Residual Dense Network used for feature extraction.
*   **Submodules:** `codes/submodules.py`
    *   *Task:* Detail the basic building blocks (convolutional layers, attention blocks) used in the network.
**Objective in Review:** Demonstrate deep understanding of *how* the neural network is constructed to learn cloud removal from satellite imagery.

## Member 2: Training Strategy & Optimization
**Responsibility:** Training Loop, Loss Functions, and Optimization.
**Key Components:**
*   **Training Script:** `codes/train_CR_kaggle.py`
    *   *Task:* Walk through the training process, epoch iteration, backpropagation, and optimizer steps.
*   **Loss Functions:** `codes/losses.py`
    *   *Task:* Explain the mathematical formulation of losses used (e.g., L1, Perceptual, FFT Loss) and their specific roles in image restoration.
*   **Checkpoint Management:** `codes/kaggle_checkpoint_helper.py`
    *   *Task:* Show how model state is saved, resumed, and managed during long training sessions on Kaggle.
**Objective in Review:** Explain the learning process and how the model is optimized to minimize errors.

## Member 3: Data Engineering & Analysis
**Responsibility:** Data Loading, Preprocessing, and Dataset Management.
**Key Components:**
*   **Data Loader:** `codes/dataloader.py`
    *   *Task:* Explain the custom logic for loading large satellite datasets, batching, and on-the-fly augmentation.
*   **Cloud Analysis:** `codes/analyze_cloud_coverage.py`
    *   *Task:* Show the statistical analysis of cloud patterns in the dataset used to filter or weight samples.
*   **Data Preparation:** `codes/generate_kaggle_data_csv.py`
    *   *Task:* Detail how the dataset manifest is created and how training/validation splits are organized.
**Objective in Review:** Demonstrate the rigorous data handling pipeline that ensures valid and high-quality input for the model.

## Member 4: Validation, Metrics & Performance
**Responsibility:** Testing, Benchmarking, and Result Analysis.
**Key Components:**
*   **Testing Scripts:** `codes/test_CR_kaggle.py` & `codes/test_baseline_kaggle.py`
    *   *Task:* Showcase the inference pipeline for generating results on unseen data.
*   **Metrics:** `codes/metrics.py`
    *   *Task:* Define and explain quantitative metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index) used to judge quality.
*   **Comparative Analysis:** `codes/test_metrics_validation.py`
    *   *Task:* Present the code that runs the direct comparison between the proposed model and baseline methods to prove superiority.
**Objective in Review:** Prove the effectiveness of the project through rigorous testing and quantitative comparison against state-of-the-art methods.
