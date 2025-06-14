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
1. Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New Research Resource for Complex Physiologic Signals. Circulation [Online]. Available: https://physionet.org/

2. Kemp B, Zwinderman AH, Tuk B, Kamphuisen HAC, Oberye JJL. (2000). Analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the EEG. IEEE Transactions on Biomedical Engineering.

3. Rechtschaffen, A., & Kales, A. (1968). A manual of standardized terminology, techniques and scoring system for sleep stages of human subjects. U.S. National Institute of Neurological Diseases and Blindness.

4. MNE-Python Documentation. https://mne.tools/stable/index.html

5. TensorFlow Documentation. https://www.tensorflow.org/

6. Smets J, Claes S, et al. (2018). Large-scale evaluation of sleep stage classification algorithms using EEG data. IEEE Journal of Biomedical and Health Informatics.

7. Tsinalis O, Matthews PM, Guo Y. (2016). Automatic Sleep Stage Scoring Using Time-Frequency Analysis and Stacked Sparse Autoencoders. Annals of Biomedical Engineering.

8. Faul F, et al. (2007). GPower 3: A flexible statistical power analysis program for the social, behavioral, and biomedical sciences*. Behavior Research Methods.

9. Abou Jaoude M, et al. (2022). Deep learning in sleep stage classification: A survey and new model. Computer Methods and Programs in Biomedicine.

10. Repository inspiration and reference:
- https://github.com/byhyu/sleep-stage-classification
- https://github.com/swayanshu/Sleep-Stage-Classification
