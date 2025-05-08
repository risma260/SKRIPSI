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
selected2 = option_menu(None, ["Dokumentasi", "Akurasi"], 
    icons=['graph-up', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")



#Halaman hasil pemodelan XGBoost
if (selected2 == 'Dokumentasi') :
    st.subheader('Dokumentasi')
    st.subheader("2.1 Load Dataset")
    st.write("Berikut merupakan code yang digunakan untuk melakukan load dataset pada python")
    code = '''df = pd.read_csv("dataset_dbd.csv")'''
    st.code(code, language="python")
    st.write("Output: ")

    data = pd.read_csv("dataset_dbd.csv")
    st.write(data) 

    st.subheader("2.2 Data Transformation")
    st.write("Berikut merupakan code yang digunakan untuk melakukan data transformation pada python")

    data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
    data['jenis_demam'] = data['jenis_demam'].map({'DSS': 2, 'DBD': 1, 'DD': 0})
        
    code = '''
    data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
    data['jenis_demam'] = data['jenis_demam'].map({'DSS': 2, 'DBD': 1, 'DD': 0})
    median_value = data['lama_dirawat'].median()
    data['kategori_lama_dirawat'] = data['lama_dirawat'].apply(lambda x: 1 if x > median_value else 0)
    '''
    st.code(code, language="python")
    st.write("Output: ", data)

    st.subheader("2.3 Imputasi Missing Value")
    st.write("Berikut merupakan code yang digunakan untuk melakukan data transformation pada python")

    # missing_values = data.isnull().sum()

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
    cols_reference = ['jenis_kelamin', 'jenis_demam', 'trombosit', 'kategori_lama_dirawat', 'hct', 'hemoglobin']

    #lakukan imputasi menggunakan function yang telah dibuat
    impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5)
    data.head()
    '''
    st.code(code, language="python")
    # st.text(missing_values)

    def impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5):
        imputer = KNNImputer(n_neighbors=n_neighbors)
        cols_reference_numeric = data[cols_reference].select_dtypes(include=['float64', 'int64']).columns.tolist()
        imputed_values = imputer.fit_transform(data[cols_reference_numeric + cols_to_impute])
        data[cols_to_impute] = imputed_values[:, len(cols_reference_numeric):]
        return data
        
    # pilih kolom yang akan dilakukan imputasi
    cols_to_impute = ['hct', 'hemoglobin']
    cols_reference = ['jenis_kelamin', 'jenis_demam', 'trombosit', 'kategori_lama_dirawat', 'hct', 'hemoglobin']

    #lakukan imputasi menggunakan function yang telah dibuat
    impute_missing_values(data, cols_to_impute, cols_reference, n_neighbors=5)

    st.write("Output: ", data)

    st.subheader("2.3 Minmax Normalization")
    st.write("Berikut merupakan code yang digunakan untuk melakukan normalisasi data pada python")

    # missing_values = data.isnull().sum()

    code = '''
    # membuat fungsi untuk imputasi missing value
    def normalize_columns(data, cols_to_normalize):
        scaler = MinMaxScaler()
        data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
        return data
        
    #pilih kolom yang akan dinormalisasi menggunakan fungsi sebelumnya
    normalize_columns(data, ['umur', 'hct', 'hemoglobin', 'trombosit'])
    '''
    st.code(code, language="python")

    def normalize_columns(data, cols_to_normalize):
        scaler = MinMaxScaler()
        data[cols_to_normalize] = scaler.fit_transform(data[cols_to_normalize])
        # joblib.dump(scaler, /content/drive/MyDrive/Task/Semester 8/Dataset/)
        return data
        
    normalize_columns(data, ['umur', 'hct', 'hemoglobin', 'trombosit'])

    st.write("Output: ", data)

    st.subheader("2.4 Menghapus Fitur")
    st.write("Berikut merupakan code yang digunakan untuk menghapus fitur yang tidak diperlukan pada python")

    code = '''
    # membuat fungsi untuk remove columns
    def remove_columns(data, columns_to_remove):
        data = data.drop(columns=columns_to_remove)
        return data

    # menghapus fitur menggunakan fungsi
    data_final = remove_columns(data, ['tgl_masuk', 'tgl_keluar', 'rm', 'lama_dirawat'])
    '''
    st.code(code, language="python")

    def remove_columns(data, columns_to_remove):
        data = data.drop(columns=columns_to_remove)
        return data
        
    data_final = remove_columns(data, ['tgl_masuk', 'tgl_keluar', 'rm', 'lama_dirawat'])
    data_final.head()

    st.write("Output: ", data_final)

    st.subheader("2.5 Membagi Dataset")
    st.write("Berikut merupakan code yang digunakan untuk membagi dataset menjadi 70% training dan 30% testing pada python")

    code = '''
    # membuat fungsi untuk remove columns
    train_data, test_data = train_test_split(data_final, test_size=0.3, random_state=42)

    # melihat hasil pembagian data
    print(train_data.shape)
    print(test_data.shape)

    '''

    train_data, test_data = train_test_split(data_final, test_size=0.3, random_state=42)

    st.code(code, language="python")

    st.write("Output: ")
    st.write("Hasil data training: ", train_data.shape)
    st.write("Hasil data test: ", test_data.shape)

    st.subheader("2.7 Metode Tunggal")
    st.subheader("2.7.1 K-Nearest Neighbord (KNN)")
    st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan KNN pada python")

    code = '''
    # menggunakan model knn pada sklearn
    knn = KNeighborsClassifier()

    # menentukan parameter yang akan di tuning
    parameters_knn = {
        'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
        'metric': ['euclidean']
    }

    # melakukan grid search pada knn
    grid_search_knn = GridSearchCV(knn, parameters_knn, cv=kf, scoring='accuracy')
    grid_search_knn.fit(X_train, y_train)

    # menampilkan hasil parameter terbaik
    print(grid_search_knn.best_score_)
    print(grid_search_knn.best_params_)

    # melakukan prediksi pada data testing menggunakan parameter terbaik
    best_model_knn = grid_search_knn.best_estimator_
    y_pred_knn = best_model_knn.predict(X_test)

    '''

    st.code(code, language="python")

    st.write("Output:")
    st.text("Hasil terbaik: 0.6509614015097565 Parameter: {'metric': 'euclidean', 'n_neighbors': 9}")


    st.subheader("2.9 Evaluasi Model")
    st.write("Berikut merupakan code yang digunakan untuk melakukan evaluasi model")

    code = '''
    #membuat function untuk evaluasi model
    def evaluate_model(y_true, y_pred):
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precission": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1-score": f1_score(y_true, y_pred)
        }

    return metrics

    #melakukan evaluasi model menggunakan function
    eval_knn  = evaluate_model(y_test, y_pred_knn)
    eval_lr   = evaluate_model(y_test, y_pred_lr)
    eval_svm  = evaluate_model(y_test, y_pred_svm)
    eval_mlp  = evaluate_model(y_test, y_pred_mlp)

    #menyimpan hasil evaluasi
    metrics_base_model = {
        "KNN" : eval_knn,
        "LR"  : eval_lr,
        "SVM" : eval_svm,
        "ANN" : eval_mlp,
    }

    '''

    table_results = pd.DataFrame(
        [
            {"Model": "KNN", "Accuracy": 0.62, "Precission": 0.44, "Recall": 0.32, "F1-Score": 0.37},
            {"Model": "LR", "Accuracy": 0.65, "Precission": 0.52, "Recall": 0.17, "F1-Score": 0.25},
            {"Model": "SVM", "Accuracy": 0.67, "Precission": 0.69, "Recall": 0.12, "F1-Score": 0.21},
            {"Model": "ANN", "Accuracy": 0.65, "Precission": 0.52, "Recall": 0.18, "F1-Score": 0.26},
        ]
    )

    st.code(code, language="python")

    st.write("Output: ")

    st.dataframe(table_results, use_container_width=True)

    st.write("Berikut merupakan code yang digunakan untuk melakukan evaluasi model ensemble")

    code = '''
        
    #melakukan evaluasi model menggunakan function
    eval_ensemble_knn = evaluate_model(y_test, y_pred_ensemble_knn)
    eval_ensemble_lr  = evaluate_model(y_test, y_pred_ensemble_lr)
    eval_ensemble_svm = evaluate_model(y_test, y_pred_ensemble_svm)
    eval_ensemble_mlp = evaluate_model(y_test, y_pred_ensemble_mlp)

    #menyimpan hasil evaluasi
    metrics_ensemble = {
        "Ensemble KNN" : eval_ensemble_knn,
        "Ensemble LR" : eval_ensemble_lr,
        "Ensemble SVM" : eval_ensemble_svm,
        "Ensemble ANN" : eval_ensemble_mlp
    }

    '''

    table_results_ensemble = pd.DataFrame(
        [
            {"Model": "KNN", "Accuracy": 0.66, "Precission": 0.65, "Recall": 0.27, "F1-Score": 0.36},
            {"Model": "LR", "Accuracy": 0.66, "Precission": 0.56, "Recall": 0.17, "F1-Score": 0.26},
            {"Model": "SVM", "Accuracy": 0.62, "Precission": 0.44, "Recall": 0.32, "F1-Score": 0.37},
            {"Model": "ANN", "Accuracy": 0.65, "Precission": 0.53, "Recall": 0.10, "F1-Score": 0.17},
        ]
    )

    st.code(code, language="python")

    st.write("Output: ")

    st.dataframe(table_results_ensemble, use_container_width=True)









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
