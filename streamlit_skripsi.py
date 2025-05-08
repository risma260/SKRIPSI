import streamlit as st
import pandas as pd
import pickle
import joblib
from streamlit_option_menu import option_menu
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# Judul web
st.title('Aplikasi Prediksi Lama Rawat Inap Pasien Demam Berdarah')

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Dokumentasi", "Prediksi"], 
    icons=['graph-up', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#Halaman hasil pemodelan XGBoost
if (selected2 == 'akurasi') :
    st.subheader('Akurasi Model')

    if 'processed_data' in st.session_state:
        df = st.session_state['processed_data']
        
        # Split data
        X = df.drop(columns=['lama_dirawat'])
        y = df['lama_dirawat']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=10, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        
         # Evaluasi model
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_rmse = math.sqrt(train_mse)
        test_rmse = math.sqrt(test_mse)

        # Menampilkan hasil evaluasi
        st.write(f"**RMSE Train Set:** {train_rmse:.4f}")
        st.write(f"**RMSE Test Set:** {test_rmse:.4f}")
        
        # Plot hasil prediksi vs aktual
        fig, ax = plt.subplots(figsize=(8,5))
        sns.lineplot(x=range(len(y_test)), y=y_test, label='Actual')
        sns.lineplot(x=range(len(y_test)), y=y_test_pred, label='Predicted')
        ax.set_title("Prediksi vs Aktual")
        ax.set_xlabel("Index")
        ax.set_ylabel("Lama Rawat")
        st.pyplot(fig)

        # Simpan model ke file pickle
        with open('model_xgboost.pkl', 'wb') as file:
            pickle.dump(best_model, file)

        st.success("Model berhasil disimpan sebagai 'model_xgboost.pkl'!")
    else:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu di halaman Preprocessing.")

# Halaman Implementasi
# Load model dan scaler dari file .joblib
xgb_model = joblib.load('xgb_best_model.joblib')           # Model hasil GridSearchCV
scaler_x = joblib.load('scaler_fitur.joblib')         # Scaler untuk fitur
scaler_y = joblib.load('scaler_target.joblib')        # Scaler untuk target

# Input data
col1, col2 = st.columns(2)

with col1:
    umur = st.number_input('Umur (tahun)', min_value=0)
    trombosit = st.number_input('Jumlah Trombosit (x10^3/μL)', min_value=0)
    hct = st.number_input('Hematokrit (HCT %)', min_value=0.0, step=0.1)

with col2:
    hemoglobin = st.number_input('Hemoglobin (HB g/dL)', min_value=0.0, step=0.1)
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
    jenis_demam = st.selectbox('Jenis Demam', ['DD', 'DBD', 'DSS'])

# Encoding fitur kategorikal
jenis_kelamin_mapping = {'Laki-laki': 0, 'Perempuan': 1}
jenis_demam_mapping = {'DD': 0, 'DBD': 1, 'DSS': 2}

jenis_kelamin_encoded = jenis_kelamin_mapping[jenis_kelamin]
jenis_demam_encoded = jenis_demam_mapping[jenis_demam]

# Tombol prediksi
if st.button('Prediksi Lama Rawat Inap'):
    try:
        # Gabungkan semua input ke dalam array
        input_data = np.array([[umur, hemoglobin, hct, trombosit, jenis_kelamin_encoded, jenis_demam_encoded]])

        # Normalisasi hanya fitur numerik (4 kolom pertama)
        input_data[:, :4] = scaler_x.transform(input_data[:, :4])

        # Prediksi
        prediksi_normal = xgb_model.predict(input_data)

        # Denormalisasi hasil prediksi
        prediksi_lama_rawat = scaler_y.inverse_transform(prediksi_normal.reshape(-1, 1))

        # Tampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        st.write(f"Perkiraan lama rawat inap: {round(prediksi_lama_rawat[0][0])} hari")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
