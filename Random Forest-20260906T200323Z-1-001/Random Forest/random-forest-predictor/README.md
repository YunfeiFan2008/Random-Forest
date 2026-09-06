# Random Forest Predictor

This project implements a Random Forest model from scratch for predicting economic and housing data. The model is designed to handle various datasets and provide accurate predictions based on the features provided.

## Project Structure

```
random-forest-predictor
├── src
│   ├── main.py               # Entry point for the application
│   ├── random_forest         # Contains the Random Forest implementation
│   │   ├── __init__.py       # Initializes the Random Forest module
│   │   ├── decision_tree.py   # Implementation of the Decision Tree class
│   │   └── random_forest.py   # Implementation of the Random Forest class
│   ├── data                  # Contains data-related functionalities
│   │   └── __init__.py       # Initializes the Data module
│   └── utils                 # Contains utility functions
│       └── __init__.py       # Initializes the Utils module
├── requirements.txt          # Lists project dependencies
└── README.md                 # Project documentation
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd random-forest-predictor
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
   Then install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application**:
   Execute the main script to load the dataset and train the Random Forest model:
   ```
   python src/main.py
   ```

## Dataset

The dataset used in this project is focused on economic and housing data. Ensure that the dataset is in the correct format and located in the appropriate directory before running the application.

## Model Functionality

- The Random Forest model is built using multiple decision trees to improve prediction accuracy.
- The implementation includes features such as bootstrapping and feature selection.
- The model can be trained on various datasets and provides predictions based on the trained trees.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.