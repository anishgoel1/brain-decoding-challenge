import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from .dataloader import DataLoader
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


class BrainDecoder:
    def __init__(self):
        self.dataloader = DataLoader()
        self.scaler = StandardScaler()
        self.models = {
            'svm': SVC(probability=True, random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'neural_net': MLPClassifier(max_iter=1000, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }


    def train_and_evaluate(self, subject_id, n_splits=5):
        X, y = self.dataloader.load_subject_data(subject_id)
        X = X.reshape(X.shape[0], -1)
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {model_name: [] for model_name in self.models}
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale the features
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            
            # Train and evaluate each model
            for model_name, model in self.models.items():
                print(f"\nTraining {model_name} - Fold {fold + 1}/{n_splits}")
                
                # Train the model
                model.fit(X_train, y_train)
                
                # Make predictions and calculate accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                results[model_name].append(accuracy)
                
                print(f"{model_name.upper()} Fold {fold + 1} Accuracy: {accuracy:.4f}")
        
        # Calculate and print mean accuracies
        mean_results = {}
        for model_name, accuracies in results.items():
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)
            mean_results[model_name] = {
                'mean_accuracy': mean_acc,
                'std_accuracy': std_acc
            }
            print(f"\n{model_name.upper()} - Mean Accuracy: {mean_acc:.4f} (±{std_acc:.4f})")
            
        return mean_results

if __name__ == "__main__":
    decoder = BrainDecoder()
    
    # Train and evaluate models for both subjects
    subjects = ["New10Subject1", "New10Subject2"]
    all_results = {}
    
    for subject in subjects:
        try:
            print(f"\nProcessing {subject}...")
            results = decoder.train_and_evaluate(subject)
            all_results[subject] = results
        except Exception as e:
            print(f"Error processing {subject}: {str(e)}")
    
    print("\nFinal Results Summary:")
    for subject, results in all_results.items():
        print(f"\n{subject}:")
        for model_name, metrics in results.items():
            print(f"{model_name}: {metrics['mean_accuracy']:.4f} (±{metrics['std_accuracy']:.4f})")