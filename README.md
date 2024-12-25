## Brain Decoding Challenge 
Training a convolutional LSTM model to predict words from fNIRS data. We reached 93% and 90% on patients 1,2 respectively.

1. **Clone the repository**
   ```bash
   git clone https://github.com/anishgoel1/brain-decoding-challenge.git
   cd brain-decoding-challenge
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python3 -m pip install 'tensorflow[and-cuda]'
   ```

4. **Train and Evaluate the Model**
   ```bash
   python main.py
   ```

## Results

### Subject 1
5-Fold Cross-Validation Results:
- Fold 1: 93.49%
- Fold 2: 93.40%
- Fold 3: 93.72%
- Fold 4: 93.58%
- Fold 5: 93.07%

**Average Test Accuracy: 93.45%**

### Subject 2
5-Fold Cross-Validation Results:
- Fold 1: 91.96%
- Fold 2: 89.14%
- Fold 3: 91.37%
- Fold 4: 91.52%
- Fold 5: 86.16%

**Average Test Accuracy: 90.03%**

## Hardware Specifications
The model was trained on:
- 1 NVIDIA L4 GPU
- 12 vCPU
- 201 GB RAM

   