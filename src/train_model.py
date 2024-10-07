import numpy as np
import pandas as pd
import pickle  # For saving and loading models

# Calculate the Gini Index for a list of class labels
def gini_index(groups, classes):
    total_samples = sum([len(group) for group in groups])
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        class_counts = np.bincount([row[-1] for row in group], minlength=len(classes))
        proportions = class_counts / size
        gini += (1.0 - np.sum(proportions ** 2)) * (size / total_samples)
    return gini

# Split a dataset based on an attribute and attribute value
def test_split(index, value, dataset):
    dataset = np.array(dataset)
    left = dataset[dataset[:, index] < value].tolist()
    right = dataset[dataset[:, index] >= value].tolist()
    return left, right

# Select the best split point for a dataset
def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))  # Unique class labels
    best_index, best_value, best_score, best_groups = None, None, float('inf'), None
    dataset = np.array(dataset)
    
    for index in range(len(dataset[0]) - 1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < best_score:
                best_index, best_value, best_score, best_groups = index, row[index], gini, groups
    
    return {'index': best_index, 'value': best_value, 'groups': best_groups}

# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

# Recursive function to split the dataset and build the tree
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])

    # Check if either group is empty
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return

    # Check for maximum depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return

    # Process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth + 1)

    # Process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth + 1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

# Save the trained decision tree using pickle
def save_model(tree, model_path):
    with open(model_path, 'wb') as f:
        pickle.dump(tree, f)
    print(f"Model saved to {model_path}")

# Make a prediction with a decision tree
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

# Load and prepare the real dataset
def load_real_data(file_path):
    data = pd.read_csv(file_path)
    dataset = data.values.tolist()
    return dataset

if __name__ == '__main__':
    # Load the preprocessed data
    data_path = r'C:\Users\ASUS\Desktop\Ayush_Sharma_A1\decision_tree_task\data\preprocessed_training_data.csv'
    dataset = load_real_data(data_path)
    print("Data loaded.")
    
    # Define max depth and min size
    max_depth = 2
    min_size = 300
    print(f"Max depth: {max_depth}, Min size: {min_size}")
    
    # Train the decision tree on the real dataset
    tree = build_tree(dataset, max_depth, min_size)
    print("Decision Tree trained on real data.")
    
    # Save the trained model
    model_path = 'decision_tree_task/models/decision_tree_model_final.pkl'
    save_model(tree, model_path)
    
    # Make a prediction on a sample row from the dataset
    sample_row = dataset[0][:-1]  # Take the first row, excluding the class label
    prediction = predict(tree, sample_row)
    print(f"Prediction for the first row: {prediction}")
