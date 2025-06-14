import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dropout, MaxPool1D, Flatten, concatenate, Dense, Reshape, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
from scipy.signal import butter, lfilter
import pandas as pd

st.set_page_config(page_title="Sleep Stage Classification")

WINDOW_SIZE = 100
Fs = 100

# Mô hình 1: CNN 3 Head
def modelcnn3head(n_classes=5):
    n_timesteps = 3000
    n_features = 1
    n_outputs = n_classes
    inputs1 = Input(shape=(n_timesteps, n_features))
    conv1 = Conv1D(64, 3, activation='relu')(inputs1)
    drop1 = Dropout(0.5)(conv1)
    pool1 = MaxPool1D(pool_size=2)(drop1)
    flat1 = Flatten()(pool1)

    inputs2 = Input(shape=(n_timesteps, n_features))
    conv2 = Conv1D(64, 5, activation='relu')(inputs2)
    drop2 = Dropout(0.5)(conv2)
    pool2 = MaxPool1D(pool_size=2)(drop2)
    flat2 = Flatten()(pool2)

    inputs3 = Input(shape=(n_timesteps, n_features))
    conv3 = Conv1D(64, 11, activation='relu')(inputs3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPool1D(pool_size=2)(drop3)
    flat3 = Flatten()(pool3)

    merged = concatenate([flat1, flat2, flat3])
    dense1 = Dense(100, activation='relu')(merged)
    outputs = Dense(n_outputs, activation='softmax')(dense1)

    model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# Mô hình 2: CNN + LSTM
def model_cnn_lstm(n_classes=5):
    inputLayer = Input(shape=(3000, 1))
    convFine = Conv1D(64, kernel_size=int(Fs/2), strides=int(Fs/16), padding='same', activation='relu')(inputLayer)
    convFine = MaxPool1D(pool_size=8, strides=8)(convFine)
    convFine = Dropout(0.5)(convFine)
    convFine = Conv1D(128, 8, padding='same', activation='relu')(convFine)
    convFine = Conv1D(128, 8, padding='same', activation='relu')(convFine)
    convFine = Conv1D(128, 8, padding='same', activation='relu')(convFine)
    convFine = MaxPool1D(pool_size=4, strides=4)(convFine)
    convFine = Flatten()(convFine)

    convCoarse = Conv1D(32, kernel_size=Fs*4, strides=int(Fs/2), padding='same', activation='relu')(inputLayer)
    convCoarse = MaxPool1D(pool_size=4, strides=4)(convCoarse)
    convCoarse = Dropout(0.5)(convCoarse)
    convCoarse = Conv1D(128, 6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(128, 6, padding='same', activation='relu')(convCoarse)
    convCoarse = Conv1D(128, 6, padding='same', activation='relu')(convCoarse)
    convCoarse = MaxPool1D(pool_size=2, strides=2)(convCoarse)
    convCoarse = Flatten()(convCoarse)

    mergeLayer = concatenate([convFine, convCoarse])
    outLayer = Dropout(0.5)(mergeLayer)
    outLayer = Reshape((1, outLayer.shape[1]))(outLayer)
    outLayer = LSTM(64, return_sequences=True)(outLayer)
    outLayer = LSTM(64, return_sequences=False)(outLayer)
    outLayer = Dense(n_classes, activation='softmax')(outLayer)

    model = Model(inputs=inputLayer, outputs=outLayer)
    model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def butter_bandpass(lowcut, highpass, fs, order=4):
    nyq = 0.5 * fs
    #       low = lowcut / nyq
    high = highpass / nyq
    b, a = butter(order, high, btype='highpass')
    return b, a

def butter_bandpass_filter(data, highpass, fs, order=4):
    b, a = butter_bandpass(0, highpass, fs, order=order)
    y = lfilter(b, a, data)
    return y 

# Load mô hình với cache
@st.cache_resource
def load_model_1():
    model = modelcnn3head()
    model.load_weights("models/model3heads.weights.h5")
    return model

@st.cache_resource
def load_model_2():
    model = model_cnn_lstm()
    model.load_weights("models/model_cnn.weights.h5")
    return model

# Giao diện Streamlit
st.title("🧠 Sleep Stage Classification App")
st.markdown("Upload một đoạn EEG tín hiệu dài 30s (~3000 mẫu điểm).")

uploaded_file = st.file_uploader("Tải file dữ liệu .npz", type=["npz"])

if uploaded_file is not None:
    try:
        npzfile = np.load(uploaded_file)
        X = npzfile['x']
        y = npzfile['y']

        st.write("✅ Dữ liệu đã tải thành công.")
        st.write(f"Shape của X: {X.shape}")
        st.write(f"Shape của y: {y.shape}")

        idx = st.slider("Chọn chỉ số mẫu để xem tín hiệu", 0, len(X)-1, 0)
        st.line_chart(X[idx].squeeze())

        # Dự đoán toàn bộ tập
        st.markdown("---")
        st.subheader("📊 Dự đoán toàn bộ dữ liệu và đánh giá mô hình")

        if X.shape[1] != 3000:
            st.warning("Dữ liệu không đúng định dạng. Mỗi mẫu phải có 3000 điểm.")
        else:
            pp_X = np.array([butter_bandpass_filter(sample, highpass=40.0, fs=100, order=4) for sample in X])

            model1 = load_model_1()
            X_multi = [pp_X, pp_X, pp_X]
            y_pred_cnn3head = model1.predict(X_multi, batch_size=64)
            y_pred_cnn3head = np.array([np.argmax(s) for s in y_pred_cnn3head])
        
            model2 = load_model_2()
            y_pred_cnn_lstm = model2.predict(pp_X, batch_size=64)
            y_pred_cnn_lstm = np.array([np.argmax(s) for s in y_pred_cnn_lstm])
            # y_true = np.array([np.argmax(s) for s in y]) if y.ndim > 1 else y
            # print(y, y_pred_cnn_lstm)

            st.subheader("📋 Classification Report (CNN 3 Head):")
            f1_cnn3head = f1_score(y, y_pred_cnn3head, average="macro")
            print(f1_cnn3head)
            st.success(f"🎯 F1 score (CNN 3 Head): `{f1_cnn3head:.4f}`")
            report1 = classification_report(y, y_pred_cnn3head, target_names=['W', 'N1', 'N2', 'N3', 'REM'], output_dict=True)
            report_df1 = pd.DataFrame(report1).transpose()
            st.dataframe(report_df1)

            st.subheader("📋 Classification Report (CNN + LSTM):")
            f1_cnn_lstm = f1_score(y, y_pred_cnn_lstm, average="macro")
            print(f1_cnn_lstm)
            st.success(f"🎯 F1 score (CNN + LSTM): `{f1_cnn_lstm:.4f}`")
            report2 = classification_report(y, y_pred_cnn_lstm, target_names=['W', 'N1', 'N2', 'N3', 'REM'], output_dict=True)
            report_df2 = pd.DataFrame(report2).transpose()
            st.dataframe(report_df2)
            
            # Hiển thị biểu đồ so sánh
            st.subheader("📊 Biểu đồ so sánh F1 Score giữa hai mô hình")
            fig, ax = plt.subplots()
            sns.barplot(x=['CNN 3 Head', 'CNN + LSTM'], y=[f1_cnn3head, f1_cnn_lstm], ax=ax)
            ax.set_ylim(0, 1)
            ax.set_ylabel('F1 Score')
            ax.set_title('So sánh F1 Score giữa hai mô hình')
            st.pyplot(fig)

            # Hiển thị biểu đồ confusion matrix cho từng mô hình
            st.subheader("📊 Confusion Matrix (CNN 3 Head)")
            fig1, ax1 = plt.subplots()
            sns.heatmap(pd.crosstab(y, y_pred_cnn3head, rownames=['True'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues', ax=ax1)
            ax1.set_title('Confusion Matrix (CNN 3 Head)')
            st.pyplot(fig1)
            st.subheader("📊 Confusion Matrix (CNN + LSTM)")
            fig2, ax2 = plt.subplots()
            sns.heatmap(pd.crosstab(y, y_pred_cnn_lstm, rownames=['True'], colnames=['Predicted']), annot=True, fmt='d', cmap='Blues', ax=ax2)
            ax2.set_title('Confusion Matrix (CNN + LSTM)')
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
