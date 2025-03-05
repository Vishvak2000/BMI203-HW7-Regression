"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
from regression import logreg, utils
import numpy as np

@pytest.fixture # learnt using AI
def logistic_regressor():
    
    return logreg.LogisticRegressor(num_feats=2)

def test_prediction(logistic_regressor):
  
    X_test = np.array([
        [1.0, 2.0],   # Should likely predict 1
        [-1.0, -2.0]  # Should likely predict 0
    ])
    
    # Set weights to create a clear LINEAR decision boundary
    logistic_regressor.W = np.array([1.0, 1.0])
    
    # Make predictions
    predictions = logistic_regressor.make_prediction(X_test)
    
    
	# assert predictions are probabilities
    assert np.all(predictions >= 0) & np.all(predictions <= 1)  # Probabilities
    assert len(predictions) == len(X_test)  # Correct number of predictions
    assert predictions[0] > 0.5 and predictions[1] < 0.5    # Correct predictions

def test_loss_function(logistic_regressor):

    y_true = np.array([1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8])  
    
    # Calculate loss
    loss = logistic_regressor.loss_function(y_true, y_pred)
    
    assert isinstance(loss, float)  # Loss is a float
    assert loss >= 0  # Loss is non-negative
    # assert loss is the correct value for CE loss
    assert np.isclose(loss, 0.144, atol=0.001)  

def test_gradient(logistic_regressor):
    # Create a small test dataset
    X_test = np.array([
        [1.0, 2.0],
        [2.0, 3.0]
    ])
    y_true = np.array([1, 0])
    
    # Set initial weights
    logistic_regressor.W = np.array([0.5, 0.5])
    
    # Calculate gradient
    gradient = logistic_regressor.calculate_gradient(y_true, X_test)
    
    # Assert gradient is correct value
    assert np.allclose(gradient, np.array([0.8, 1.2]), atol=0.1)  # Gradient is correct
    

def test_training(logistic_regressor):
    X_train= np.array([
        [1.0, 2.0],
        [2.0, 3.0]
    ])
    y_true = np.array([1, 0])
    
	# create validation set
    X_val = np.array([
		[3.0, 4.0],
		[4.0, 5.0]
	])
    y_val = np.array([1, 0])
    
    
    initial_weights = logistic_regressor.W.copy()
    
    # Train the model
    logistic_regressor.train_model(X_train, y_true, X_val, y_val)
    
    # Assertions
    assert not np.array_equal(initial_weights, logistic_regressor.W)  # Weights changed
    assert len(logistic_regressor.loss_hist_train) > 0  # Training loss history is not empty