import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import os
import gdown

# === Hàm tải model từ Google Drive ===
@st.cache_resource
def load_model_from_drive(model_name, file_id):
    model_path = f"models/{model_name}"
    if not os.path.exists(model_path):
        os.makedirs("models", exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return tf.keras.models.load_model(model_path)

# === Tải 2 model từ Drive ===
model_cnn3head = load_model_from_drive("multi_head_cnn_sleep.h5", "ID_MODEL_1")
model_cnnltsm = load_model_from_drive("best_cnnlstm_model.h5", "ID_MODEL_2")

# === Giao diện ===
st.title("🛌 Sleep Stage Classification")
st.markdown("Tải dữ liệu tín hiệu EEG (.npz) để dự đoán giai đoạn giấc ngủ với 2 mô hình và so sánh kết quả.")

uploaded_file = st.file_uploader("📂 Tải lên file dữ liệu (.npz)", type=["npz"])

if uploaded_file is not None:
    try:
        npz = np.load(uploaded_file)
        st.write("📂 Các biến có trong file:", npz.files)

        if 'x' in npz and 'y' in npz:
            x = npz['x']
            y_true = npz['y']

            if len(x.shape) != 3 or x.shape[1:] != (3000, 1):
                st.error("❌ Dữ liệu phải có shape (batch_size, 3000, 1).")
            else:
                labels = ['Wake', 'N1', 'N2', 'N3', 'REM']

                # === Mô hình 1: CNN 3 Head ===
                st.subheader("🔹 Mô hình 1: CNN 3 Head")
                x3 = [x, x, x]
                y_pred1 = np.argmax(model_cnn3head.predict(x3), axis=1)
                acc1 = np.mean(y_pred1 == y_true)
                st.success(f"🎯 Accuracy (CNN 3 Head): {acc1 * 100:.2f}%")
                st.bar_chart(pd.DataFrame(np.bincount(y_pred1, minlength=5), index=labels, columns=["Số lượng"]))

                # === Mô hình 2: CNN-LSTM ===
                st.subheader("🔹 Mô hình 2: CNN-LSTM")
                try:
                    y_pred2 = np.argmax(model_cnnltsm.predict([x,x,x]), axis=1)
                    acc2 = np.mean(y_pred2 == y_true)
                    st.success(f"🎯 Accuracy (CNN-LSTM): {acc2 * 100:.2f}%")
                    st.bar_chart(pd.DataFrame(np.bincount(y_pred2, minlength=5), index=labels, columns=["Số lượng"]))

                    # So sánh
                    st.subheader("📊 So sánh phân bố giữa 2 mô hình")
                    df_compare = pd.DataFrame({
                        'CNN 3 Head': np.bincount(y_pred1, minlength=5),
                        'CNN-LSTM': np.bincount(y_pred2, minlength=5)
                    }, index=labels)
                    st.bar_chart(df_compare)

                    # Confusion Matrix
                    st.subheader("🧮 Confusion Matrix")
                    for name, y_pred, cm_color in [("CNN 3 Head", y_pred1, 'Blues'), ("CNN-LSTM", y_pred2, 'OrRd')]:
                        fig, ax = plt.subplots()
                        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
                        sns.heatmap(cm, annot=True, fmt='d', cmap=cm_color, xticklabels=labels, yticklabels=labels, ax=ax)
                        ax.set_title(f"{name} - Confusion Matrix")
                        st.pyplot(fig)

                    # Mức độ giống nhau
                    match_rate = np.mean(y_pred1 == y_pred2)
                    st.info(f"🧩 Hai mô hình giống nhau ở {match_rate*100:.2f}% số epoch.")

                    # Báo cáo phân loại
                    st.subheader("📑 Báo cáo phân loại")
                    st.markdown("**CNN 3 Head**")
                    st.dataframe(pd.DataFrame(classification_report(y_true, y_pred1, target_names=labels, output_dict=True)).transpose())
                    st.markdown("**CNN-LSTM**")
                    st.dataframe(pd.DataFrame(classification_report(y_true, y_pred2, target_names=labels, output_dict=True)).transpose())

                except Exception as e2:
                    st.error(f"❌ Lỗi khi chạy CNN-LSTM: {e2}")
        else:
            st.error("❌ File không chứa 'x' và 'y'.")
    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
