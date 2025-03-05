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
import seaborn as sns


# Judul web
st.title('Aplikasi Prediksi Lama Rawat Inap Pasien Demam Berdarah')

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Preprocessing", "akurasi", "Implementasi"], 
    icons=['house', 'filter', 'graph-up', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#halaman Data
if (selected2 == 'Data') :
    
    # Upload file CSV atau Excel
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Baca file sesuai ekstensi
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Simpan dataset ke session_state agar bisa digunakan di halaman lain
            st.session_state['uploaded_data'] = df
    
            # Tampilkan data yang telah diunggah
            st.write("### Data yang diunggah:")
            st.dataframe(df)
    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")


# Fungsi untuk memproses data
@st.cache_data
def preprocess_data(df):
    # Memilih fitur yang digunakan
    selected_features = ['jenis_kelamin','umur', 'jenis_demam', 'trombosit','hemoglobin', 'hct', 'lama_dirawat']
    df = df[selected_features]
    
    # Encoding fitur kategorikal
    encoder = LabelEncoder()
    df['jenis_kelamin'] = encoder.fit_transform(df['jenis_kelamin'])
    df['jenis_demam'] = encoder.fit_transform(df['jenis_demam'])
    
    # Normalisasi data
    scaler = MinMaxScaler()
    df[['jenis_kelamin','umur', 'jenis_demam', 'trombosit', 'hemoglobin', 'hct']] = scaler.fit_transform(df[['jenis_kelamin','umur', 'jenis_demam', 'trombosit', 'hemoglobin', 'hct']])
    
    # Imputasi KNN
    imputer = KNNImputer(n_neighbors=5)
    df.iloc[:, :] = imputer.fit_transform(df)
    
    return df, scaler

# Halaman Preprocessing
if selected2 == 'Preprocessing':
    st.subheader('Preprocessing Data')

    # Pastikan ada data yang sudah diupload
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
        st.write("Data Awal:")
        st.dataframe(df)

        df_processed, scaler = preprocess_data(df)

        # Simpan hasil preprocessing dan scaler ke session_state
        st.session_state["processed_data"] = df_processed
        joblib.dump(scaler, "scaler.pkl")  # Simpan scaler ke file
        st.success("Scaler berhasil disimpan sebagai 'scaler.pkl'!")

        st.write("### Data Setelah Preprocessing:")
        st.dataframe(df_processed)

    else:
        st.warning("Silakan upload dataset terlebih dahulu di halaman Data.")

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
if selected2 == "Implementasi":
    st.subheader("Implementasi")

    # Load model
    try:
        with open("model_xgboost.pkl", "rb") as file:
            dbd_model = pickle.load(file)
        st.success("Model berhasil dimuat.")
    except FileNotFoundError:
        st.error("Model belum tersedia. Silakan jalankan training di halaman Akurasi terlebih dahulu.")
        dbd_model = None

    # Load scaler
    try:
        scaler = joblib.load("scaler.pkl")
        st.success("Scaler berhasil dimuat.")
    except FileNotFoundError:
        st.error("Scaler belum tersedia. Silakan jalankan preprocessing di halaman Preprocessing terlebih dahulu.")
        scaler = None

    if dbd_model and scaler:
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
                # Format input untuk prediksi
                input_data = np.array([[umur, trombosit, hct, hemoglobin, jenis_kelamin_encoded, jenis_demam_encoded]])
                
                # Prediksi dengan model
                prediksi_lama_rawat = dbd_model.predict(input_data)

                # Menampilkan hasil prediksi
                st.subheader('Hasil Prediksi')
                st.write(f"Perkiraan lama rawat inap: {round(prediksi_lama_rawat[0])} hari")

            except Exception as e:
                st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

    else:
        st.warning("Model belum tersedia. Silakan jalankan training di halaman Akurasi terlebih dahulu.")
