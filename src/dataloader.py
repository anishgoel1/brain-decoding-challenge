import os
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


class DataLoader:
    """Data loader for fNIRS brain decoding challenge."""

    def __init__(self):
        self.data_dir = "data"
        self.data_channels = 84  

    def load_subject_data(self, subject_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and aggregate all trials across all dates for a subject.

        The fNIRS data consists of 84 data channels (42 HBO + 42 HBR)

        Args:
            subject_id: ID of the subject (e.g., 'New10Subject1')

        Returns:
            Tuple of (X, y) where:
                X: numpy array of shape (n_trials, n_channels)
                y: numpy array of labels (0-10, where 0 is rest)
        """
        subject_path = os.path.join(self.data_dir, subject_id)
        if not os.path.exists(subject_path):
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")
            
        date_folders = [f for f in os.listdir(subject_path) 
                       if os.path.isdir(os.path.join(subject_path, f))]

        X_all, y_all = [], []

        for date in date_folders:
            try:
                # Load data and labels for this date
                data = np.load(os.path.join(subject_path, date, f"{date}PreprocessedData.npy"))
                labels = np.load(os.path.join(subject_path, date, f"{date}Labels.npy"), 
                               allow_pickle=True)

                # Ensure labels match data length
                if len(labels) > len(data):
                    labels = labels[:len(data)]

                for trial_idx, trial_channel_data in enumerate(data):
                    # Aggregate over timepoints (e.g., compute mean) to get a 84-dimensional vector
                    X_all.append(np.mean(trial_channel_data, axis=1))
                    y_all.append(int(float(labels[trial_idx, 2])))

            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue

        if not X_all:
            raise ValueError(f"No valid trials found for subject {subject_id}")

        # Convert to numpy arrays
        X = np.array(X_all)
        y = np.array(y_all)

        print(f"\nLoaded {len(X)} trials for subject {subject_id}")
        
        return X, y


if __name__ == "__main__":
    loader = DataLoader()
    try:
        X1, y1 = loader.load_subject_data("New10Subject1")
        print(f"New10Subject1: {X1.shape}, {y1.shape}")
        
        X2, y2 = loader.load_subject_data("New10Subject2")
        print(f"New10Subject2: {X2.shape}, {y2.shape}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")