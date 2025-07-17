# Simple Audio-to-Text (ASR) Deep Learning Pipeline

This project demonstrates a simple yet functional audio transcription (speech-to-text) pipeline built from scratch using PyTorch. It uses the Common Voice dataset for training and supports inference on external audio files like podcasts or downloaded speech.

## Project Structure

- `models/`: Contains a custom Conv1D + GRU-based ASR model.
- `utils/`: Dataset loading, preprocessing, and training utilities.
- `scripts/train.py`: Entry-point script to train the model.
- `scripts/infer.py`: Script to transcribe an external audio file.
- `inference/`: Inference logic and model loading.
- `results/report.txt`: Training report and observations.
- `ASR_Example.ipynb`: Jupyter Notebook walkthrough.

## Features

- Built from scratch using PyTorch.
- Uses torchaudio and Common Voice dataset.
- CTC Loss for training.
- Inference-ready with pretrained model.
- Code structured for readability and reuse.

## Requirements

```bash
pip install torch torchaudio matplotlib
```

## Usage

### 1. Train the Model
```bash
python scripts/train.py
```

### 2. Run Inference
```bash
python scripts/infer.py
```

## Notes

- Only a sample of the dataset is used for faster training.
- Accuracy can improve significantly with more data and training time.
