import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Conv1D,
    MaxPooling1D,
    Dropout,
    BatchNormalization,
    Input,
)
import tensorflow as tf


class DataLoader:
    def __init__(self, max_timepoints, data_dir="data"):
        """
        Initialize the DataLoader class.
        :param max_timepoints: Max number of timepoints per trial to consider.
        :param data_dir: Directory containing the data.
        :param standardize: Whether to standardize the data.
        :param window_size: Size of the sliding window.
        :param stride: Stride of the sliding window.
        """
        self.data_dir = data_dir
        self.max_timepoints = max_timepoints
        self.data_channels = 84
        self.standardize = True
        self.window_size = 10
        self.stride = 5

    def create_windows_and_labels(self, data, labels):
        data = data.reshape(-1, self.data_channels, self.max_timepoints)
        windows, window_labels = [], []

        for trial, label in zip(data, labels):
            for i in range(0, self.max_timepoints - self.window_size + 1, self.stride):
                window = trial[:, i : i + self.window_size]
                windows.append(window)
                window_labels.append(label)

        return np.array(windows), np.array(window_labels)

    def load_subject_data(self, subject_id):
        """
        Load the data for a specific subject.
        :param subject_id: ID of the subject.
        :return: Data and labels after all additional processing.
        """
        subject_path = os.path.join(self.data_dir, subject_id)
        if not os.path.exists(subject_path):
            raise FileNotFoundError(f"Subject directory not found: {subject_path}")

        X_all, y_all = [], []
        for date_folder in os.listdir(subject_path):
            folder_path = os.path.join(subject_path, date_folder)
            if os.path.isdir(folder_path):
                try:
                    # use preprocessed data from the folders
                    data_path = os.path.join(
                        folder_path, f"{date_folder}PreprocessedData.npy"
                    )
                    labels_path = os.path.join(folder_path, f"{date_folder}Labels.npy")

                    data = np.load(data_path)
                    # the labels are in the format [timestamp, duration, label]
                    labels = (
                        np.load(labels_path, allow_pickle=True)[:, 2]
                        .astype(float)
                        .astype(int)
                    )

                    for trial, label in zip(data, labels):
                        trial = self._pad_or_trim_trial(trial)
                        X_all.append(trial.flatten())
                        y_all.append(label)

                except Exception as e:
                    print(f"Error processing folder {date_folder}: {e}")

        X_all, y_all = np.array(X_all), np.array(y_all)

        if self.standardize:
            X_all = self._standardize_data(X_all)

        return self.create_windows_and_labels(X_all, y_all)

    def _pad_or_trim_trial(self, trial):
        """
        Pad or trim a trial to the specified maximum timepoints.
        :param trial: Trial data.
        :return: Padded or trimmed trial.
        """
        if trial.shape[1] > self.max_timepoints:
            return trial[:, : self.max_timepoints]
        elif trial.shape[1] < self.max_timepoints:
            padding = np.zeros((trial.shape[0], self.max_timepoints - trial.shape[1]))
            return np.hstack((trial, padding))
        return trial

    def _standardize_data(self, data):
        """
        Standardize the data (z-score normalization).
        :param data: Data to be standardized.
        :return: Standardized data.
        """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        return (data - mean) / (std + 1e-8)


def create_conv_lstm_model(X, num_classes):
    """
    Create a Convolutional LSTM model.
    :param X: Input data.
    :param num_classes: Number of classes.
    :return: Convolutional LSTM model.
    """
    model = Sequential(
        [
            Input(shape=(X.shape[1], X.shape[2])),
            Conv1D(64, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            Conv1D(128, 3, activation="relu"),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.3),
            LSTM(128, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(64, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    return model


def train_and_evaluate_conv_lstm_cv(X, y, subject_name):
    """
    Train and evaluate a Convolutional LSTM model using 5-fold cross-validation.
    :param X: Input data.
    :param y: Labels.
    :param subject_name: Name of the subject.
    :return: Average test accuracy.
    """
    y_cat = tf.keras.utils.to_categorical(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_cat[train_idx], y_cat[test_idx]

        model = create_conv_lstm_model(X, len(np.unique(y)))

        tf.keras.mixed_precision.set_global_policy("mixed_float16")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        model.fit(X_train, y_train, epochs=75, batch_size=64, verbose=0)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        fold_accuracies.append(test_accuracy)
        print(f"Fold {fold + 1} Test Accuracy: {test_accuracy * 100:.2f}%")

    avg_accuracy = np.mean(fold_accuracies)
    print(
        f"\n{subject_name} 5-Fold Cross-Validation Average Test Accuracy: {avg_accuracy * 100:.2f}%"
    )
    return avg_accuracy


if __name__ == "__main__":
    """Main script to load data, perform 5-fold cross-validation, and evaluate models."""
    loaders = [DataLoader(max_timepoints=93), DataLoader(max_timepoints=68)]
    patients = ["New10Subject1", "New10Subject2"]
    for patient, loader in zip(patients, loaders):
        print(f"Loading data for {patient}...")
        X, y = loader.load_subject_data(patient)
        print(f"Performing 5-Fold Cross-Validation for {patient}...")
        avg_accuracy = train_and_evaluate_conv_lstm_cv(X, y, patient)
