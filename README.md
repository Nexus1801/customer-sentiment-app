# Customer Sentiment Classification Shiny Application

## Overview
This comprehensive R Shiny application implements a complete data science lifecycle for customer sentiment analysis using various machine learning classification algorithms. The application allows users to interactively explore data, preprocess features, train multiple models, evaluate performance, and make predictions.

## Features

### 1. **Data Science Lifecycle Implementation**
- **Business Understanding**: Predict customer sentiment for improved satisfaction
- **Data Understanding**: Interactive exploration and visualization
- **Data Preparation**: Automated preprocessing with user-configurable options
- **Modeling**: Multiple classification algorithms with hyperparameter tuning
- **Evaluation**: Comprehensive metrics and visualizations
- **Deployment**: Interactive prediction interface

### 2. **Machine Learning Algorithms**
- **Random Forest**: Ensemble learning with configurable trees and features
- **Support Vector Machine (SVM)**: Kernel-based classification
- **Decision Tree**: Interpretable rule-based model
- **Naive Bayes**: Probabilistic classifier

### 3. **Interactive Features**
- Upload custom datasets or use demo data
- Adjust preprocessing parameters (train/test split, normalization)
- Configure model hyperparameters in real-time
- Compare multiple models simultaneously
- Make single or batch predictions
- Download prediction results

## Installation

### Required R Packages

```r
# Install required packages
install.packages(c(
  "shiny",
  "shinydashboard",
  "DT",
  "ggplot2",
  "dplyr",
  "caret",
  "randomForest",
  "e1071",
  "rpart",
  "rpart.plot",
  "plotly",
  "corrplot",
  "reshape2"
))
```

### Running the Application

1. **Download the dataset**:
   - Visit: https://www.kaggle.com/datasets/kundanbedmutha/customer-sentiment-dataset
   - Download `customer_sentiment_dataset.csv`

2. **Launch the application**:
```r
# Method 1: Run directly from R/RStudio
library(shiny)
runApp("path/to/app.R")

# Method 2: If in the same directory
shiny::runApp()
```

3. **Or use the demo data** by clicking "Load Demo Data" button in the application

## Usage Guide

### Step 1: About & Methodology
- Read the project overview and methodology
- Upload your dataset or load demo data
- Review the data science lifecycle implementation

### Step 2: Data Exploration
- **Dataset Overview**: View summary statistics and structure
- **Data Table**: Browse the raw data
- **Target Distribution**: Visualize class balance
- **Missing Values**: Identify data quality issues
- **Feature Distributions**: Explore individual variables
- **Correlation Matrix**: Understand feature relationships

### Step 3: Data Preprocessing
- **Configure Parameters**:
  - Adjust train/test split ratio (50-90%)
  - Enable/disable feature normalization
  - Choose missing value imputation
  - Select target variable
- **Apply Preprocessing**: Click to prepare data for modeling
- **Review Results**: Check split distribution and feature summary

### Step 4: Model Training
- **Select Algorithm**: Choose from RF, SVM, DT, or NB
- **Configure Hyperparameters**:
  - **Random Forest**: Number of trees, variables per split
  - **SVM**: Kernel type, cost parameter
  - **Decision Tree**: Max depth, minimum split
  - **Naive Bayes**: (No hyperparameters)
- **Train Model**: Click to start training
- **View Performance**: Check training summary and feature importance

### Step 5: Model Evaluation
- **Performance Metrics**: View accuracy, precision, recall
- **Confusion Matrix**: Analyze classification breakdown
- **Classification Report**: Detailed per-class metrics
- **Model Comparison**: Train and compare all algorithms

### Step 6: Predictions
- **Single Prediction**:
  - Enter feature values
  - Click "Predict Sentiment"
  - View prediction and confidence scores
- **Batch Predictions**:
  - Upload CSV file
  - Generate predictions
  - Download results