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
st.title('Aplikasi Prediksi Lama Rawat Inap Pasien Demam Berdarah Berdasarkan Hasil Diagnostik Laboratorium Menggunakan Metode XGBoost')

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Dokumentasi", "Prediksi"], 
    icons=['graph-up', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")



#Halaman hasil pemodelan XGBoost
if (selected2 == 'Dokumentasi') :
    st.subheader('Dokumentasi')
    st.subheader("1. Load Dataset")
    st.write("Dataset dimuat dari file CSV yang berisi data pasien demam berdarah, mencakup informasi klinis seperti umur, jenis kelamin, hasil laboratorium, dan lama rawat inap.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        df = pd.read_csv("dataset_dbd.csv")
        '''
        st.code(code, language="python")
        st.write("Output: ")
    
        data = pd.read_csv("dataset_dbd.csv")
    st.write(data) 
    
    st.subheader("2. Data Cleaning")
    st.write("Proses ini dilakukan untuk menghapus fitur yang tidak digunakan dalam pemodelan.")
    
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        def remove_columns(dataset, columns_to_remove):  
            data = dataset.drop(columns=columns_to_remove)
            return data
        data = remove_columns(data, ['rm', 'tgl_masuk', 'tgl_keluar'])
        '''
        st.code(code, language="python")
    
    st.write("Output:")
    data = pd.read_csv("data_removed.csv")
    st.write(data)


    st.subheader("3. Data Transformation")
    st.write("Transformasi data dilakukan untuk mengonversi fitur kategorikal menjadi numerik.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):    
        code = '''
        # Encoding untuk fitur 'jenis_kelamin'
        le_jenis_kelamin = LabelEncoder()
        data['jenis_kelamin'] = le_jenis_kelamin.fit_transform(data['jenis_kelamin'])
        
        # Custom encoding untuk fitur 'jenis_demam'
        jenis_demam_encode = {'DD': 0, 'DBD': 1, 'DSS': 2}
        data['jenis_demam'] = data['jenis_demam'].map(jenis_demam_encode)
        '''
        st.code(code, language="python")
    st.write("Output: ")
    data = pd.read_csv("data_encoded.csv")
    st.write(data) 
    

    st.subheader("4. Imputasi Missing Value")
    st.write("Tahap ini bertujuan untuk menangani nilai yang hilang (missing value) dalam dataset menggunakan metode KNN")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # membuat fungsi untuk imputasi missing value
        def impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5):
            imputer = KNNImputer(n_neighbors=n_neighbors)
            cols_reference_numeric = data[cols_reference].select_dtypes(include=['float64', 'int64']).columns.tolist()
            imputed_values = imputer.fit_transform(data[cols_reference_numeric + cols_to_impute])
            data[cols_to_impute] = imputed_values[:, len(cols_reference_numeric):]
            return data
            
        # pilih kolom yang akan dilakukan imputasi
        cols_to_impute = ['hct', 'hemoglobin']
        cols_reference = ['jenis_kelamin', 'jenis_demam', 'umur', 'trombosit', 'lama_dirawat', 'hct', 'hemoglobin']
    
        #lakukan imputasi menggunakan function yang telah dibuat
        impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5)
        '''
        st.code(code, language="python")
        # st.text(missing_values)

    st.write("Output: ")
    data = pd.read_csv("data_imputed.csv")
    st.write(data) 
    

    st.subheader("5. Normalisasi Data")
    st.write("Normalisasi menggunakan MinMaxScaler dilakukan untuk menyelaraskan skala data agar setiap fitur berada dalam rentang yang sama.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # Inisialisasi MinMaxScaler untuk fitur dan target
        scaler_x = MinMaxScaler()
        scaler_y = MinMaxScaler()
    
        # Mengubah format koma menjadi titik dan konversi ke float
        data['hemoglobin'] = data['hemoglobin'].replace({',': '.'}, regex=True).astype(float)
        data['hct'] = data['hct'].replace({',': '.'}, regex=True).astype(float)
        data['trombosit'] = pd.to_numeric(data['trombosit'], errors='coerce')
    
        # Tentukan fitur numerik yang ingin dinormalisasi
        fitur_numerik = ['umur', 'hemoglobin', 'hct', 'trombosit']
    
        # Normalisasi fitur numerik
        data[fitur_numerik] = scaler_x.fit_transform(data[fitur_numerik])
    
        # Normalisasi target (lama_dirawat)
        data['lama_dirawat'] = scaler_y.fit_transform(data[['lama_dirawat']])
        '''
        st.code(code, language="python")

    st.write("Output: ")
    data = pd.read_csv("data_normalized.csv")
    st.write(data) 
    
    st.subheader("6. Split Data")
    st.write("Dataset dibagi menjadi data latih (80%) dan data uji (20%) untuk keperluan pelatihan dan evaluasi model.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # Pisahkan fitur dan target
        X = data.drop(['lama_dirawat'], axis=1)  # Hapus kolom target dari fitur
        y = data['lama_dirawat']  # Target asli
    
        # Bagi data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
        '''
        st.code(code, language="python")

    st.write("Output: ")
    st.write("Jumlah data training: 678")
    st.write("Jumlah data testing: 170")

    st.subheader("7. Mencari Hyperparameter Terbaik")
    st.write("Tuning hyperparameter dilakukan menggunakan GridSearch untuk memperoleh kombinasi parameter terbaik bagi model XGBoost.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # Inisialisasi model XGBoost Regressor
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
        
        # Menyusun kombinasi hyperparameter yang akan diuji
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2, 0.3],     
            'max_depth': [3, 4, 5, 6, 7],               
            'n_estimators': [100, 200, 300, 400],       
            'subsample': [0.6, 0.7, 0.8, 0.9]           
        }
    
        # Membuat objek GridSearchCV untuk mencari kombinasi terbaik
        grid_search = GridSearchCV(
            estimator=xgb_model,                       
            param_grid=param_grid,                    
            cv=5,                                      
            n_jobs=-1,                                 
        )
    
        # Melatih GridSearchCV dengan data training
        start_time = time.time()                       
        grid_search.fit(X_train, y_train)              
        elapsed_time = time.time() - start_time        
    
        '''
        st.code(code, language="python")

    st.write("Output:")
    st.text("Best Hyperparameters: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 100, 'subsample': 0.6}")
    st.text("Waktu komputasi: 209.75 detik (3.50 menit)")


    st.subheader("8. Evaluasi Model")
    st.write("Evaluasi dilakukan untuk mengukur performa model terhadap data uji menggunakan metrik mse dan rmse.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # Ambil model terbaik dari GridSearchCV
        xgb_best = grid_search.best_estimator_
        
        # Prediksi pada data testing
        y_test_pred = xgb_best.predict(X_test)
        
        # Evaluasi model
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        
            '''
        st.code(code, language="python")

    st.write("Output: ")
    st.text("RMSE: 0.1107")
    
    st.subheader("9. Denormalisasi")
    st.write("Denormalisasi dilakukan untuk mengembalikan nilai hasil prediksi yang sebelumnya telah dinormalisasi, sehingga hasil prediksi dapat digunakan atau dibandingkan dengan data asli.")
    # Menambahkan collapsible expander
    with st.expander("📄 Lihat kode program"):
        code = '''
        # Denormalisasi hasil prediksi dan target menggunakan min_target_asli dan max_target_asli
        y_test_asli = y_test * (max_target_asli - min_target_asli) + min_target_asli
        y_test_pred_asli = y_test_pred * (max_target_asli - min_target_asli) + min_target_asli
        y_test_pred_rounded = np.round(y_test_pred_asli) # Membulatkan hasil
        df_hasil = pd.DataFrame({
            'Target Asli': y_test_asli,
            'Prediksi': y_test_pred_rounded
        })
        
            '''
        st.code(code, language="python")

    st.write("Output: ")
    data = pd.read_csv("data_denorm.csv")
    st.write(data) 
    





# Halaman Implementasi
#Halaman hasil pemodelan XGBoost
if (selected2 == 'Prediksi') :
    st.subheader('Prediksi')
    
    # Load model dan scaler dari file .joblib
    xgb_model = joblib.load('xgb_best_model.joblib')           # Model hasil GridSearchCV
    scaler_x = joblib.load('scaler_fitur.joblib')         # Scaler untuk fitur
    scaler_y = joblib.load('scaler_target.joblib')        # Scaler untuk target
    
    # Input data
    col1, col2 = st.columns(2)
    
    with col1:
        umur = st.number_input('Umur (tahun)', min_value=0)
        trombosit = st.number_input('Trombosit (x10^3/μL)', min_value=0)
        hct = st.number_input('Hematokrit (%)', min_value=0.0, step=0.1)

    with col2:
        hemoglobin = st.number_input('Hemoglobin (g/dL)', min_value=0.0, step=0.1)
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
