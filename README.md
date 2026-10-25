# Speech Emotion Recognition

Classifies the emotion in a speech recording (neutral, happy, sad, angry, ...) from acoustic features extracted with `librosa`. Built as Task 2 of my Machine Learning internship.

## Highlights
- **Features (per frame):** 40 MFCC + 12 Chroma + 64 log-Mel bands = 116 features x 130 frames per 3-second clip
- **Three models:** Random Forest and RBF-SVM on aggregated statistics, or a Keras **CNN + LSTM** on the full feature sequence
- **Leakage-safe evaluation:** speaker-independent train / validation / test split; scaler fitted on training data only; augmentation applied to training files only
- **Datasets:** RAVDESS, TESS, any folder of `.wav` files, or a built-in synthetic demo to verify the code runs
- **Inference:** `predict_emotion("clip.wav")` returns the label and class probabilities

## Project Structure
```
Speech_Emotion_Recognition/
├── src/emotion_recognition.py   # full pipeline + predict_emotion()
├── data/                        # put datasets here (git-ignored)
├── models/                      # saved model + model_meta.json
├── results/                     # confusion matrix, reports, training curves
├── notebooks/
├── requirements.txt
└── README.md
```

## Setup (Python 3.10+)
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
TensorFlow is only required for `--model cnn_lstm`.

## Dataset
**RAVDESS (recommended):** download *Audio_Speech_Actors_01-24* (Zenodo / Kaggle) and unzip so that files sit under `data/RAVDESS/Actor_01/ ... Actor_24/`. Labels are read from the file names (`03-01-05-01-02-01-12.wav` -> emotion 05 = angry, actor 12).

**TESS:** unzip into `data/TESS/` (labels read from file names such as `OAF_back_angry.wav`).

**Your own files:** `data/custom/<emotion>/*.wav`, run with `--dataset folder`. Name files `spk01_xxx.wav` to enable a speaker-independent split.

**Demo (no download):** `--dataset demo` generates synthetic speech-like tones. It only checks that the pipeline runs; its accuracy says nothing about real speech.

## Run
```bash
# 1. smoke test, no download needed
python src/emotion_recognition.py --dataset demo --model rf --no-show

# 2. real data: classical baseline
python src/emotion_recognition.py --dataset ravdess --model rf

# 3. real data: deep model with augmentation
python src/emotion_recognition.py --dataset ravdess --model cnn_lstm --augment

# 4. more emotion classes
python src/emotion_recognition.py --dataset ravdess --model rf --emotions neutral calm happy sad angry fearful disgust surprised

# 5. predict one file with the last trained model
python src/emotion_recognition.py --predict path/to/clip.wav
```
```python
import sys; sys.path.append("src")
from emotion_recognition import predict_emotion
predict_emotion("path/to/clip.wav")
# {'emotion': 'sad', 'confidence': 0.98, 'probabilities': {'sad': 0.98, 'neutral': 0.02, ...}}
```

## Feature Extraction Logic
1. Load at 22,050 Hz mono, trim leading/trailing silence (30 dB), pad or crop to 3 s
2. STFT with `n_fft=2048`, `hop_length=512` (130 frames)
3. Per frame: **MFCC** (vocal-tract timbre), **Chroma** (pitch-class energy, a proxy for intonation), **log-Mel** (spectral energy / loudness distribution)
4. Standardise with `StandardScaler` fitted on the training set only
5. Random Forest / SVM use the **mean, std and max** of every feature over time (348 values); the CNN-LSTM uses the full 130 x 116 sequence

## Evaluation Method
- Split by **speaker** (about 60 / 15 / 20 % train / validation / test), so test voices were never seen in training. Random file-level splits inflate scores because the same actor says the same sentences in every emotion.
- The validation set drives early stopping (CNN-LSTM) and is reported for all models; the test set is used once at the end.
- Reported: per-class Precision / Recall / F1, Accuracy, macro-F1, and a confusion matrix (`results/confusion_matrix.png`).
- Seeds are fixed (42). TensorFlow results can still vary slightly between machines.

## Results
Run the commands above and copy the numbers from `results/metrics_summary.csv` and `results/classification_report.csv`.

| Model | Dataset / emotions | Test Accuracy | Macro-F1 |
|-------|--------------------|--------------:|---------:|
| Random Forest | RAVDESS, 4 classes | _fill after running_ | _fill_ |
| SVM (RBF) | RAVDESS, 4 classes | _fill after running_ | _fill_ |
| CNN + LSTM (augmented) | RAVDESS, 4 classes | _fill after running_ | _fill_ |

Insert `results/confusion_matrix.png` and `results/training_history.png` here after training.

## Limitations
- Acted emotions in studio conditions: expect a drop on spontaneous speech
- RAVDESS has only 24 actors, so test results vary with which actors land in the test split
- Single language (English) and a limited emotion set

## License
MIT
