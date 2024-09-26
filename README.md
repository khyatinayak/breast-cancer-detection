# Breast Cancer Detection

This project is a machine learning-based approach for predicting breast cancer using a classification model. The model achieves high accuracy by analyzing input features extracted from breast cancer diagnostic data. 

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model](#model)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Breast cancer is one of the most common types of cancer in women, and early detection can significantly improve survival rates. This project leverages machine learning algorithms to classify breast cancer as benign or malignant based on features like cell size, shape, and other characteristics from digitized images of breast mass.

The model is trained on the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29), achieving a high accuracy of **97%**.

## Dataset
The dataset used in this project contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. The features describe characteristics of the cell nuclei present in the image.

- **Features**: 30 numeric attributes (e.g., radius, texture, perimeter, area, smoothness)
- **Target**: Two classes - Benign (0) and Malignant (1)
- **Source**: UCI Machine Learning Repository - [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

## Installation
To run this project locally, you need Python 3.x and the following libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`

### Steps:
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/breast-cancer-detection.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Once installed, you can run the project using the provided Jupyter Notebook or a Python script.

### To run the model:
1. **Open Jupyter Notebook**:
   ```bash
   jupyter notebook breast_cancer_detection.ipynb
   ```
2. **Run the Script**:
   ```bash
   python breast_cancer_detection.py
   ```

### Data Preprocessing
The dataset is preprocessed using techniques like scaling, missing value handling, and splitting into training and testing sets.

### Model Training and Evaluation
After preprocessing, the model is trained on the training dataset and evaluated on the test dataset.

## Results
The model achieves an accuracy of **97%**, with the following key metrics:
- **Precision**: 0.98
- **Recall**: 0.96
- **F1-Score**: 0.97

These results demonstrate the model's high performance in correctly identifying both benign and malignant cases.

## Contributing
Contributions are welcome! If you find any bugs or want to contribute enhancements, please feel free to submit a pull request or open an issue.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
