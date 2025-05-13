import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

tf.compat.v1.reset_default_graph()

# Load mô hình
@st.cache_resource
def load_model_1():
    return tf.keras.models.load_model("models/multi_head_cnn_sleep.h5")

@st.cache_resource
def load_model_2():
    return tf.keras.models.load_model("D:/Study/DH/MangCamBien/final-sleep-stage-classification/models/best_cnnlstm_model.h5")

model_cnn3head = load_model_1()
model_cnnltsm = load_model_2()

# Giao diện chính
st.title("🛌 Sleep Stage Classification")
st.markdown("Tải dữ liệu tín hiệu EEG (.npz) để dự đoán giai đoạn giấc ngủ với 2 mô hình và so sánh kết quả.")

# Upload file
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

                # 1️⃣ Mô hình CNN 3 Head
                st.subheader("🔹 Mô hình 1: CNN 3 Head")
                x3 = [x, x, x]
                pred_1 = model_cnn3head.predict(x3)
                y_pred1 = np.argmax(pred_1, axis=1)
                acc1 = np.mean(y_pred1 == y_true)
                st.success(f"🎯 Accuracy (CNN 3 Head): {acc1 * 100:.2f}%")

                # Bar chart với st.bar_chart
                counts1 = np.bincount(y_pred1, minlength=5)
                df_counts1 = pd.DataFrame(counts1, index=labels, columns=["Số lượng"])
                st.bar_chart(df_counts1)

                # 2️⃣ Mô hình CNN-LSTM
                st.subheader("🔹 Mô hình 2: CNN-LSTM")
                try:
                    model_cnnltsm.summary()
                    pred_2 = model_cnnltsm.predict([x,x,x])
                    y_pred2 = np.argmax(pred_2, axis=1)
                    acc2 = np.mean(y_pred2 == y_true)
                    st.success(f"🎯 Accuracy (CNN-LSTM): {acc2 * 100:.2f}%")

                    counts2 = np.bincount(y_pred2, minlength=5)
                    df_counts2 = pd.DataFrame(counts2, index=labels, columns=["Số lượng"])
                    st.bar_chart(df_counts2)

                    # 🔄 So sánh phân bố
                    st.subheader("📊 So sánh phân bố giữa 2 mô hình")
                    df_compare = pd.DataFrame({
                        'CNN 3 Head': counts1,
                        'CNN-LSTM': counts2
                    }, index=labels)
                    st.bar_chart(df_compare)

                except Exception as e2:
                    st.error(f"❌ Lỗi khi chạy mô hình CNN-LSTM: {e2}")
                    st.write("Chi tiết lỗi:", str(e2))

                if 'y_pred2' in locals():
                    # Confusion matrix
                    st.subheader("🧮 Confusion Matrix")

                    fig_cm1, ax_cm1 = plt.subplots()
                    cm1 = confusion_matrix(y_true, y_pred1, labels=[0,1,2,3,4])
                    sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues',
                                xticklabels=labels, yticklabels=labels, ax=ax_cm1)
                    ax_cm1.set_title("CNN 3 Head - Confusion Matrix")
                    ax_cm1.set_xlabel("Predicted")
                    ax_cm1.set_ylabel("True")
                    st.pyplot(fig_cm1)

                    fig_cm2, ax_cm2 = plt.subplots()
                    cm2 = confusion_matrix(y_true, y_pred2, labels=[0,1,2,3,4])
                    sns.heatmap(cm2, annot=True, fmt='d', cmap='OrRd',
                                xticklabels=labels, yticklabels=labels, ax=ax_cm2)
                    ax_cm2.set_title("CNN-LSTM - Confusion Matrix")
                    ax_cm2.set_xlabel("Predicted")
                    ax_cm2.set_ylabel("True")
                    st.pyplot(fig_cm2)

                    # So sánh dự đoán
                    st.subheader("🔍 Mức độ giống nhau giữa 2 mô hình")
                    match_rate = np.mean(y_pred1 == y_pred2)
                    st.info(f"🧩 Hai mô hình dự đoán giống nhau ở {match_rate*100:.2f}% số epoch.")

                    # Báo cáo phân loại
                    st.subheader("📑 Báo cáo phân loại")

                    report1 = classification_report(y_true, y_pred1, target_names=labels, output_dict=True, zero_division=0)
                    report2 = classification_report(y_true, y_pred2, target_names=labels, output_dict=True, zero_division=0)

                    st.markdown("**CNN 3 Head**")
                    st.dataframe(pd.DataFrame(report1).transpose())

                    st.markdown("**CNN-LSTM**")
                    st.dataframe(pd.DataFrame(report2).transpose())
        else:
            st.error("❌ File không chứa 'x' và 'y'.")

    except Exception as e:
        st.error(f"❌ Lỗi khi xử lý file: {e}")
        st.write("Chi tiết lỗi:", str(e))
