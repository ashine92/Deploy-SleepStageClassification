# 🧠 Sleep Stage Classification from Single-Channel EEG

This project aims to classify human sleep stages (Wake, N1, N2, N3, REM) using deep learning models on EEG signals from the PhysioNet Sleep-EDF dataset.

## 📌 Features

- Preprocessing EEG data (bandpass filtering, 30s epoching)
- Two Deep Learning models:
  - `CNN 3-Head` with multi-kernel convolution
  - `CNN + LSTM` with fine/coarse temporal resolution
- Evaluation: F1-score, Confusion Matrix
- Streamlit App for interactive visualization and inference

## 🧾 Dataset

- Source: [Sleep-EDF Expanded from PhysioNet](https://physionet.org/content/sleep-edfx/1.0.0/)
- Channel: `Fpz-Cz` (EEG)
- Sampling Rate: 100 Hz
- Sleep stages: Wake (W), N1, N2, N3, REM

## 🧮 Models

### 1. CNN 3 Head

Three parallel convolutional branches with kernel sizes [3, 5, 11]. Outputs are concatenated before final classification.

### 2. CNN + LSTM

Combines local and global temporal features using CNN and two-layer LSTM.

## 🔧 Installation

```bash
git clone https://github.com/yourusername/sleep-stage-classification.git
cd sleep-stage-classification
pip install -r requirements.txt
```

## Results
### Classification Report
![image](https://github.com/user-attachments/assets/0d303ec8-3592-4bc4-8b4d-886c945ca6be)

Figure 1. Classification Report (CNN 3 Head)

![image](https://github.com/user-attachments/assets/e6adb784-1f92-4ad3-add9-291feb080619)

Figure 2. Classification Report (CNN-LSTM)

### Confusion Matrix
![image](https://github.com/user-attachments/assets/7f53a55b-c3bf-44de-a970-8652269d1ed6)

![image](https://github.com/user-attachments/assets/d8c9052a-9589-4e4c-8f7e-0b4576d376bb)


## 📚 References
- https://github.com/byhyu/sleep-stage-classification
- https://github.com/swayanshu/Sleep-Stage-Classification
