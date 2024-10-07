import pandas as pd
import numpy as np

def preprocess_classification_data():
    # Load the data
    data_path = r'C:\Users\ASUS\Desktop\Ayush_Sharma_A1\decision_tree_task\data\fraud_train.csv'
    classification_data = pd.read_csv(data_path)

    # Drop high cardinality columns (ID-like features)
    classification_data.drop(columns=['nameOrig', 'nameDest'], inplace=True)

    # Handle missing numerical values by filling them with the median
    for column in classification_data.select_dtypes(include=[np.number]).columns:
        classification_data[column] = classification_data[column].fillna(classification_data[column].median())

    # Handle missing categorical values by filling them with the mode
    for column in classification_data.select_dtypes(include=['object']).columns:
        classification_data[column] = classification_data[column].fillna(classification_data[column].mode()[0])

    # Encode categorical variables manually using one-hot encoding
    for column in classification_data.select_dtypes(include=['object']).columns:
        dummies = pd.get_dummies(classification_data[column], prefix=column, drop_first=True)
        classification_data = pd.concat([classification_data, dummies], axis=1)
        classification_data.drop(column, axis=1, inplace=True)

    # Ensure normalization only applies to numerical columns (exclude boolean)
    numeric_columns = classification_data.select_dtypes(include=[np.number]).columns
    classification_data[numeric_columns] = (classification_data[numeric_columns] - classification_data[numeric_columns].min()) / (classification_data[numeric_columns].max() - classification_data[numeric_columns].min())

    # Save the preprocessed data
    preprocessed_path = r'C:\Users\ASUS\Desktop\Ayush_Sharma_A1\decision_tree_task\data\preprocessed_training_data.csv'
    classification_data.to_csv(preprocessed_path, index=False)

    print("Classification data preprocessing complete.")

# Run the preprocessing when the script is executed
if __name__ == '__main__':
    preprocess_classification_data()
