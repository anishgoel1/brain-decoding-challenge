import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from dataloader import DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

def preprocess_data(X: list) -> np.ndarray:
    """
    Preprocess the input data by flattening and normalizing.
    
    Args:
        X: List of numpy arrays with shape (84, timepoints)
    Returns:
        Preprocessed and flattened numpy array
    """
    X_np = np.array(X)
    num_samples = X_np.shape[0]
    flattened_X = X_np.reshape(num_samples, -1)
    return flattened_X

def main():
    # Load data
    loader = DataLoader()
    X, y = loader.load_subject_data("New10Subject1")
    
    # Preprocess
    X_preprocessed = preprocess_data(X)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_preprocessed)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(random_state=42))
    ])
    
    # Define parameter grid for optimization
    param_grid = {
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.1, 0.01],
        'svm__degree': [2, 3, 4]  # for poly kernel
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Print best parameters and score
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation accuracy: {:.2f}%".format(grid_search.best_score_ * 100))
    
    # Evaluate on test set
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
