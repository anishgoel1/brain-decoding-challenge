# Telepathic Technologies' Language of the Brain Decoding Challenge

## Overview
This project focuses on decoding individual words imagined by experimental subjects from their brain data using fNIRS (functional Near InfraRed Spectroscopy) neuroimaging data.

## Project Goals
- Subject 1 Classification Accuracy Goal: 80%
- Subject 2 Classification Accuracy Goal: 70%

## Data Description
- Data Type: fNIRS neuroimaging data
- Sampling Rate: 6.1Hz
- Data Channels: 84 (42 actual channels × 2 measurements)
  - Oxygenated-hemoglobin (hbo)
  - De-oxygenated-hemoglobin (hbr)
- Experiment Duration: ~10 minutes per subject
- Stimulus Duration: 15 seconds
- Rest Period: 15 seconds
- Words Used (NATO phonetic alphabet):
  - Bravo, Echo, Golf, Hotel, Kilo
  - November, Papa, Tango, Uniform, Whiskey

## Project Structure

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
   ```

4. **Data**

   Place the `New10Subject1` and `New10Subject2` folders inside the `data/` directory.