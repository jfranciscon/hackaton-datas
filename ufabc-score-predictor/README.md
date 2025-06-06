# UFABC Score Predictor

This project aims to predict the minimum score required for approval at UFABC using machine learning techniques. The model is trained on historical data and can provide insights into the scoring requirements for prospective students.

## Project Structure

```
ufabc-score-predictor
├── src
│   ├── data_preparation.py       # Data loading and preprocessing
│   ├── train_model.py            # Model training and evaluation
│   ├── predict.py                 # Prediction functionality
│   └── utils
│       └── __init__.py           # Utility functions
├── requirements.txt               # Project dependencies
├── .gitignore                     # Files to ignore in version control
└── README.md                      # Project documentation
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ufabc-score-predictor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Prepare the data by running the `data_preparation.py` script:
   ```
   python src/data_preparation.py
   ```

2. Train the model using the `train_model.py` script:
   ```
   python src/train_model.py
   ```

3. Make predictions with the trained model using the `predict.py` script:
   ```
   python src/predict.py
   ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features.

## License

This project is licensed under the MIT License.