# EEG Classification for Metabolic State Detection: A Comprehensive Analysis

## Overview 

![Overview](images/pipeline.png)

## Dataset Understanding

I'm working with brain wave data collected using EEG (Electroencephalography). This method captures brain activity by placing electrodes on the scalp using the international 10-20 system - but with an extended 64-electrode configuration instead of the standard 19 electrodes.

Each electrode captures 5 different brain wave frequency bands:

- **Alpha (8-13 Hz)**: Associated with relaxed alertness, meditation
- **Beta (13-30 Hz)**: Associated with normal waking consciousness, active thinking
- **Delta (0.5-4 Hz)**: Associated with deep sleep
- **Theta (4-8 Hz)**: Associated with drowsiness, meditation, creativity
- **Gamma (30-100 Hz)**: Associated with higher cognitive processes, peak concentration

The band power values represent the intensity of activity in each frequency range at each electrode location. The target variable (0 or 1) represents different metabolic states of the brain at that particular moment.

### Dataset Characteristics

- **Samples**: 40 participants (balanced dataset with 20 samples in each class)
- **Features**: 320+ features (64 electrodes × 5 frequency bands)
- **Challenge**: High-dimensional data with limited samples (classic n << p problem)

## Task 1: Traditional Classification Approaches

My task was to classify this EEG data using three different algorithms:

1. Linear Regression (Logistic Regression for classification)
2. Support Vector Machines (SVM) with different kernels
3. k-Nearest Neighbors (KNN)

For each model, I needed to report accuracy, precision, recall, and F1 score.

### Data Preparation

Given the small dataset size but high feature dimensionality, I used stratified train-test split to maintain class balance in both training and validation sets. This is essential for reliable performance estimation when working with balanced but small datasets.

![EEG electrode distribution](images/channel_distribution.png)
### Analysis Results

#### Linear Regression (Logistic Regression)

```
{'Accuracy': 0.25, 'Precision': 0.25, 'Recall': 0.25, 'F1 Score': 0.25}
```

To check whether the data was linearly separable, I plotted it using PCA to reduce to 2 dimensions:

![PCA](images/pca.png)

The visualization confirmed my suspicion - the data isn't linearly separable, which explains the poor performance of the logistic regression model.

#### Support Vector Machines (SVM)

##### Linear Kernel

```
Train:  {'Accuracy': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'F1 Score': 1.0}
Validation:  {'Accuracy': 0.375, 'Precision': 0.3333333333333333, 'Recall': 0.25, 'F1 Score': 0.2857142857142857}
```

The linear SVM showed clear signs of overfitting - perfect on training data but poor on validation. This reinforced that linear boundaries won't work well on this data.

##### Polynomial Kernel

```
Train:  {'Accuracy': 0.5625, 'Precision': 0.5333333333333333, 'Recall': 1.0, 'F1 Score': 0.6956521739130435}
Validation:  {'Accuracy': 0.5, 'Precision': 0.5, 'Recall': 1.0, 'F1 Score': 0.6666666666666666}
```

I tried different polynomial degrees (even up to degree=100) but performance remained consistent around 50% accuracy.

##### RBF Kernel

```
Train:  {'Accuracy': 0.59375, 'Precision': 0.5517241379310345, 'Recall': 1.0, 'F1 Score': 0.7111111111111111}
Validation:  {'Accuracy': 0.375, 'Precision': 0.4285714285714285, 'Recall': 0.75, 'F1 Score': 0.5454545454545454}
```

##### Sigmoid Kernel

```
Train:  {'Accuracy': 0.59375, 'Precision': 0.5517241379310345, 'Recall': 1.0, 'F1 Score': 0.7111111111111111}
Validation:  {'Accuracy': 0.5, 'Precision': 0.5, 'Recall': 1.0, 'F1 Score': 0.6666666666666666}
```

I also experimented with different C values (regularization parameter) from 0.1 to 100, but they didn't significantly improve performance:

```
C = 0.1 to 100.0
Validation: {'Accuracy': 0.5, 'Precision': 0.5, 'Recall': 1.0, 'F1 Score': 0.6666666666666666}
```

This suggests complex, non-linear relationships in the data that these kernels struggle to model effectively.

#### k-Nearest Neighbors (KNN)

I tested KNN with different k values (1-9) using the ball_tree algorithm:

```
n_neighbours = 1
Train:  {'Accuracy': 1.0, 'Precision': 1.0, 'Recall': 1.0, 'F1 Score': 1.0}
Validation:  {'Accuracy': 0.25, 'Precision': 0.25, 'Recall': 0.25, 'F1 Score': 0.25}

n_neighbours = 3
Train:  {'Accuracy': 0.8125, 'Precision': 0.7777777777777778, 'Recall': 0.875, 'F1 Score': 0.8235294117647058}
Validation:  {'Accuracy': 0.125, 'Precision': 0.2, 'Recall': 0.25, 'F1 Score': 0.22222222222222222}

n_neighbours = 7
Train:  {'Accuracy': 0.65625, 'Precision': 0.7272727272727273, 'Recall': 0.5, 'F1 Score': 0.5925925925925926}
Validation:  {'Accuracy': 0.25, 'Precision': 0.25, 'Recall': 0.25, 'F1 Score': 0.25}
```

KNN performed even worse than SVM, with several k values (4, 5, 6, 8, 9) completely failing on the validation set (0% for precision, recall, and F1). This is likely due to the curse of dimensionality - KNN struggles in high-dimensional spaces where distance measures become less meaningful.

### Interpretation & Reasoning

Looking at the results across all models, I found:

1. **Non-linear patterns**: All linear models (logistic regression, linear SVM) performed poorly, confirming the data isn't linearly separable.
    
2. **Modest improvement with non-linear methods**: Polynomial and sigmoid kernels reached 50% accuracy, better than linear methods but still limited.
    
3. **Overfitting across models**: All models showed significant drops between training and validation performance.
    
4. **Curse of dimensionality**: With 320+ features but only 40 samples, the models struggled. This particularly affected KNN, which relies on meaningful distance metrics.
    
5. **Complexity of brain data**: EEG signals contain complex, non-linear relationships that these basic models couldn't fully capture.
    

The best performing model was SVM with polynomial/sigmoid kernels, reaching 50% accuracy. This suggests the data contains non-linear patterns, but the high dimensionality and small sample size severely limited performance.

## Task 2: Feature Selection and Dimensionality Reduction

To address the high-dimensionality problem identified in Task 1, I explored three different approaches to identify the most informative features:

### Top 5 Features Identified by Each Method

- **PCA (Principal Component Analysis):**
    
    - Features: `['beta17', 'theta1', 'delta6', 'delta47', 'delta45']`
    - _Note:_ PCA doesn't directly select original features. These represent the original features with the highest _absolute loadings_ (contribution) across the top 5 principal components.
- **UFS (Univariate Feature Selection using `SelectKBest` with `f_classif`):**
    
    - Features: `['alpha10', 'gamma23', 'gamma34', 'gamma38', 'gamma45']`
- **RFE (Recursive Feature Elimination):**
    
    - Features: `['alpha41', 'beta43', 'theta21', 'theta29', 'theta62']`

### Why the Selected Features Differ Across Methods

The selected features are **completely different** across the three methods due to their fundamentally different approaches to feature evaluation:

1. **UFS (`SelectKBest` with `f_classif`):**
    
    - **Method**: Evaluates each feature _individually_ using ANOVA F-values to measure the relationship between single features and the target variable.
    - **Characteristics**: Ignores feature interactions and potential redundancy. Selects features purely based on their individual discriminative power.
2. **RFE (Recursive Feature Elimination):**
    
    - **Method**: Iteratively trains a model on all features, removes the least important feature(s), and repeats until reaching the desired number of features.
    - **Characteristics**: Considers features _collectively_ through the lens of the chosen model. Selects features that work well _together_ to optimize model performance.
3. **PCA (Principal Component Analysis):**
    
    - **Method**: Transforms features into uncorrelated components ordered by variance explained, not directly selecting features.
    - **Characteristics**: Focuses on capturing data variance rather than target prediction. Feature importance is determined by contribution to principal components rather than predictive power.

### Insights from Feature Selection Analysis

1. **Frequency band patterns**: The gamma band appears prominently in UFS results, suggesting individual gamma frequencies might have stronger univariate relationships with the metabolic state.
    
2. **Electrode location diversity**: Features from different brain regions were selected across methods, indicating that information relevant to metabolic state classification may be distributed throughout the brain.
    
3. **Feature engineering opportunity**: The lack of consensus between methods suggests complex relationships in the data that might benefit from more sophisticated feature engineering approaches.
    

## Novel Approach: Graph Neural Networks for EEG Classification

Given the spatial relationships between EEG electrodes and the complex patterns in the data, I explored Graph Neural Networks (GNNs) as a potentially more suitable approach.

### Why GNNs are Appropriate for EEG Data:

1. **Spatial relationships**: The electrode montage shows clear spatial structure. Traditional algorithms treat each feature independently, but EEG electrodes have spatial connections that GNNs can leverage.
    
2. **Complex patterns**: With 64 electrodes and 5 frequency bands, GNNs can capture patterns across both spatial and frequency domains.
    
3. **Limited sample size**: With only 40 samples but 320 features, GNNs can incorporate the spatial structure as an inductive bias to potentially improve generalization.
    

### GNN Implementation Approach:

1. **Graph construction**:
    
    - **Nodes**: Each electrode (1-64) becomes a node
    - **Node features**: For each node, used the 5 frequency band values
    - **Edges**: Connected electrodes based on their physical adjacency in the EEG montage
2. **Model architecture**:
    
    - Graph Convolutional Networks (GCN) layers
    - Batch normalization for training stability
    - Global pooling to aggregate information across the graph
    - Dropout for regularization

### GNN Results and Limitations:

Despite the theoretical advantages, the GNN approach faced similar limitations as traditional methods due to the extremely small dataset size. The model showed:

1. **Overfitting**: Perfect or near-perfect training accuracy but limited validation performance
2. **Instability**: High variance in results across different train-test splits
3. **Limited generalization**: Unable to significantly outperform the best traditional methods

## Conclusions and Future Directions

The comprehensive analysis across traditional machine learning methods and graph neural networks revealed several important insights:

1. **Data scarcity is the primary limitation**: With only 40 samples and 320+ features, all models struggled with generalization regardless of their sophistication.
    
2. **Non-linear methods show promise**: The best results came from non-linear approaches (SVMs with non-linear kernels), suggesting complex relationships in the data.
    
3. **Spatial relationships matter**: The structure of EEG data suggests that approaches incorporating spatial information (like GNNs) have theoretical advantages, though they need more data to realize this potential.
    

### Next Steps:

1. **Data augmentation**: Implement EEG-specific data augmentation techniques to artificially increase the sample size.
    
2. **Transfer learning**: Leverage pre-trained models on larger EEG datasets and fine-tune on this specific task.
    

### Final Assessment:

The project demonstrates that even with sophisticated methods like GNNs, the fundamental challenge of EEG classification with limited data remains. The most promising path forward involves either acquiring more data or leveraging external knowledge through transfer learning approaches.
