import os
import numpy as np
from typing import Tuple


class DataLoader:
    """Data loader to generate datasets for the brain decoding challenge."""

    def __init__(self):
        self.data_dir = "data"
        self.data_channels = 84

    def load_subject_data(self, subject_id: str) -> Tuple[list, list]:
        """Load and aggregate all trials across all dates for a subject.

        Args:
            subject_id: ID of the subject (e.g., 'New10Subject1')

        Returns:
            Tuple of (X, y) where:
                X: list of trials, where each trial is a numpy array of shape (n_channels, n_timepoints)
                y: numpy array of labels
        """
        subject_path = os.path.join(self.data_dir, subject_id)
        if not os.path.exists(subject_path):
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        date_folders = [
            f
            for f in os.listdir(subject_path)
            if os.path.isdir(os.path.join(subject_path, f))
        ]

        X_all, y_all = [], []

        for date in date_folders:
            try:
                data_path = os.path.join(
                    subject_path, date, f"{date}PreprocessedData.npy"
                )
                labels_path = os.path.join(subject_path, date, f"{date}Labels.npy")

                data = np.load(data_path)
                labels = np.load(labels_path, allow_pickle=True)

                # Truncate labels to match data length
                # I noticed the labels file to have an extra rest session
                if len(labels) > len(data):
                    labels = labels[: len(data)]

                # Convert float labels to integers
                labels = labels[:, 2].astype(float).astype(int)

                for trial_idx in range(len(data)):
                    X_all.append(np.array(data[trial_idx]))
                    y_all.append(labels[trial_idx])

            except Exception as e:
                print(f"Error processing date {date}: {str(e)}")
                continue

        return X_all, np.array(y_all)


if __name__ == "__main__":
    loader = DataLoader()
    try:
        X1, y1 = loader.load_subject_data("New10Subject1")
        X2, y2 = loader.load_subject_data("New10Subject2")

    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
