import numpy as np
import pandas as pd
import pickle  # Import pickle to load the model

# Function to calculate accuracy
def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

# Function to calculate precision
def precision_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    predicted_positive = np.sum(y_pred == 1)
    return true_positive / predicted_positive if predicted_positive > 0 else 0

# Function to calculate recall
def recall_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    actual_positive = np.sum(y_true == 1)
    return true_positive / actual_positive if actual_positive > 0 else 0

# Function to calculate F1 Score
def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Function to make predictions using the decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def evaluate_and_save_predictions(file_path):
    # Load the preprocessed data
    data = pd.read_csv(file_path)
    
    # Set the correct target column
    target_column = 'isFraud'  # Update with the actual target column name
    
    # Split data into features (X) and target (y)
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # Load the pickled decision tree model
    with open('decision_tree_task/models/decision_tree_model_final.pkl', 'rb') as f:
        tree = pickle.load(f)
    
    # Make predictions
    y_pred = [predict(tree, row) for row in X.values]
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    
    # Print the evaluation metrics in the output terminal
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    
    # Save the evaluation metrics to a file
    with open('decision_tree_task/results/train_metrics.txt', 'w') as f:
        f.write("Decision Tree Metrics:\n")
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"F1 Score: {f1:.2f}\n")
    
    print("Evaluation metrics saved successfully.")
    
    # Save the predictions to a CSV file without a header as per the instructions
    np.savetxt('decision_tree_task/results/train_predictions.csv', y_pred, delimiter=',', fmt='%d', header='', comments='')
    
    print("Predictions saved successfully in train_predictions.csv.")

if __name__ == '__main__':
    file_path = 'decision_tree_task/data/preprocessed_training_data.csv'
    evaluate_and_save_predictions(file_path)
