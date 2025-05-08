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
        code = '''df = pd.read_csv("./assets/resource/dengue_fever_los_dataset.csv")'''
        st.code(code, language="python")
        st.write("Output: ")

        data = pd.read_csv("./assets/resource/dengue_fever_los_dataset.csv")
        st.write(data) 

        st.subheader("2.2 Data Transformation")
        st.write("Berikut merupakan code yang digunakan untuk melakukan data transformation pada python")

        data['jenis_kelamin'] = data['jenis_kelamin'].map({'Laki-laki': 1, 'Perempuan': 0})
        data['jenis_demam'] = data['jenis_demam'].map({'DSS': 2, 'DBD': 1, 'DD': 0})
        median_value = data['lama_dirawat'].median()
        data['kategori_lama_dirawat'] = data['lama_dirawat'].apply(lambda x: 1 if x > median_value else 0)
        
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

        st.subheader("2.7.2 Logistic Regression (LR)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan LR pada python")

        code = '''
        # menggunakan model lr pada sklearn
        lr = LogisticRegression()

        # menentukan parameter yang akan di tuning
        parameters_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        # melakukan grid search pada lr
        grid_search_lr = GridSearchCV(lr, parameters_lr, cv=kf, scoring='accuracy')
        grid_search_lr.fit(X_train, y_train)

        # menampilkan hasil parameter terbaik
        print(grid_search_lr.best_score_)
        print(grid_search_lr.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_lr = grid_search_lr.best_estimator_
        y_pred_lr = best_model_lr.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil terbaik: 0.6373735935051987 Paremeter: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}")

        st.subheader("2.7.3 Support Vector Machine (SVM)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan SVM pada python")

        code = '''
        # menggunakan model svm pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_svm = GridSearchCV(svm, parameters_svm, cv=kf, scoring='accuracy')
        grid_search_svm.fit(X_train, y_train)

        # menampilkan hasil parameter terbaik
        print(grid_search_svm.best_score_)
        print(grid_search_svm.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_svm = grid_search_svm.best_estimator_
        y_pred_svm = best_model_svm.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6458481697763852 Parameter: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}")

        st.subheader("2.7.3 Artificial Neural Network (ANN)")
        st.write("Berikut merupakan code yang digunakan untuk modeling data menggunakan ANN pada python")

        code = '''
        # menggunakan model ann pada sklearn
        mlp = MLPClassifier()

        # menentukan parameter yang akan di tuning
        parameters_mlp = {
            'max_iter': [100, 500, 1000],
            'learning_rate_init'   : [0.0001, 0.001, 0.01, 0.1],
            'activation': ['tanh', 'relu', 'logistic'],
            'hidden_layer_sizes' : [(i,) for i in range(1, 11)]
        }

        # melakukan grid search pada ann
        grid_search_mlp = GridSearchCV(mlp, parameters_mlp, cv=kf, scoring='accuracy')
        grid_search_mlp.fit(X_train, y_train)
        

        # menampilkan hasil parameter terbaik
        print(grid_search_mlp.best_score_)
        print(grid_search_mlp.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_mlp = grid_search_mlp.best_estimator_
        y_pred_mlp = best_model_mlp.predict(X_test)

        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.644195983478137 Parameter: {'activation': 'relu', 'hidden_layer_sizes': (4,), 'learning_rate_init': 0.1, 'max_iter': 1000}")

        st.subheader("2.8 Metode Ensemble Stacking")
        st.write("Untuk membuat model ensemble stacking langkah pertama adalah menyimpan masing-masing hasil prediksi pada model.")

        # table_data_desc = pd.DataFrame(
        #     [
        #         {"Data": "Prediksi validasi metode tunggal", "Keterangan": "Prediksi validasi metode tunggal pada masing-masing fold saat traning disimpan untuk digunakan pelatihan pada model Stacking"},
        #         {"Data": "Prediksi testing metode tunggal", "Keterangan": "Prediksi testing metode tunggal pada saat testing disimpan untuk digunakan testing pada model Stacking"},
        #     ]
        # )

        st.write("Keterangan")
        st.write("Prediksi validasi metode tunggal: Prediksi validasi metode tunggal pada masing-masing fold saat traning disimpan untuk digunakan pelatihan pada model Stacking")
        st.write("Prediksi testing metode tunggal: Prediksi testing metode tunggal pada saat testing disimpan untuk digunakan testing pada model Stacking")

        st.write("Berikut merupakan code yang digunakan untuk mengumpulkan hasil prediksi pada data validasi metode tunggal")

        code = '''
        # menyimpan beberapa model yang digunakan pada metode tunggal
        models = [
            ('knn_model_optimize', KNeighborsClassifier, grid_search_knn),
            ('lr_model_optimize', LogisticRegression, grid_search_lr),
            ('svm_model_optimize', SVC, grid_search_svm),
            ('mlp_model_optimize', MLPClassifier, grid_search_mlp)
        ]

        # menyiapkan list untuk hasil prediksi dan label aktual
        predictions = {model[0]: [] for model in models}
        y_val_actuals = []
        test_val_indexs = []

        for fold, (train_index, test_index) in enumerate(kf.split(X_train, y_train), 1):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            for model_name, model_class, grid_search in models:
                model = model_class(**grid_search.best_params_)
                model.fit(X_train_fold, y_train_fold)
                y_pred_val = model.predict(X_val_fold)
                predictions[model_name].extend(y_pred_val)

            y_val_actuals.extend(y_val_fold.values)
            test_val_indexs.extend(test_index)
        
        #hasil prediksi disimpan pada dataframe
        results_validation = pd.DataFrame({
            'test_index': test_val_indexs,
            'y_pred_knn': predictions['knn_model_optimize'],
            'y_pred_lr': predictions['lr_model_optimize'],
            'y_pred_svm': predictions['svm_model_optimize'],
            'y_pred_mlp': predictions['mlp_model_optimize'],
            'y_actual': y_val_actuals
        })

        '''

        st.code(code, language="python")

        st.write("Berikut merupakan code yang digunakan untuk mengumpulkan hasil prediksi pada data testing metode tunggal")

        code = '''
        # data testing sebelumnya sudah disimpan pada variabel y_pred, kemudian sekarang dikumpulkan menjadi 1
        results_testing = pd.DataFrame({
            'y_pred_knn'  : y_pred_knn,
            'y_pred_lr'   : y_pred_lr,
            'y_pred_svm'  : y_pred_svm,
            'y_pred_mlp'  : y_pred_mlp,
            'y_actual'    : y_test
        })
        '''

        st.code(code, language="python")

        st.subheader("2.7.1 Meta-Classifier KNN")
        st.write("Untuk membuat KNN sebagai meta-classifier langkah awalnya adalah menghapus y_pred_knn dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_knn dari dataset training
        ensemble_train_knn = results_validation.drop('y_pred_knn', axis=1)

        # menghapus y_pred_knn dari dataset testing
        ensemble_test_knn = results_testing.drop('y_pred_knn', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_knn, y_ensemble_train_knn = ensemble_train_knn.drop(['y_actual', 'test_index'], axis=1), ensemble_train_knn['y_actual']
        X_ensemble_test_knn, y_ensemble_test_knn = ensemble_test_knn.drop('y_actual', axis=1), ensemble_test_knn['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        knn = KNeighborsClassifier()

        # menentukan parameter yang akan di tuning
        parameters_knn = {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15, 17, 19],
            'metric': ['euclidean']
        }

        # melakukan grid search pada knn
        grid_search_ensemble_knn = GridSearchCV(knn, parameters_knn, cv=kf, scoring='accuracy')
        grid_search_ensemble_knn.fit(X_ensemble_train_knn, y_ensemble_train_knn)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_knn.best_score_)
        print(grid_search_ensemble_knn.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_knn = grid_search_ensemble_knn.best_estimator_
        y_pred_ensemble_knn = best_model_ensemble_knn.predict(X_ensemble_test_knn)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6390827517447657 Parameter: {'metric': 'euclidean', 'n_neighbors': 3}")

        st.subheader("2.7.2 Meta-Classifier Logistic Regression (LR)")
        st.write("Untuk membuat LR sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_lr dari dataset training
        ensemble_train_lr = results_validation.drop('y_pred_lr', axis=1)

        # menghapus y_pred_lr dari dataset testing
        ensemble_test_lr = results_testing.drop('y_pred_lr', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_lr, y_ensemble_train_lr = ensemble_train_lr.drop(['y_actual', 'test_index'], axis=1), ensemble_train_lr['y_actual']
        X_ensemble_test_lr, y_ensemble_test_lr = ensemble_test_lr.drop('y_actual', axis=1), ensemble_test_lr['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        lr = LogisticRegression()

        # menentukan parameter yang akan di tuning
        parameters_lr = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']
        }

        # melakukan grid search pada lr
        grid_search_ensemble_lr = GridSearchCV(lr, parameters_lr, cv=kf, scoring='accuracy')
        grid_search_ensemble_lr.fit(X_ensemble_train_lr, y_ensemble_train_lr)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_lr.best_score_)
        print(grid_search_ensemble_lr.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_lr = grid_search_ensemble_lr.best_estimator_
        y_pred_ensemble_lr = best_model_ensemble_lr.predict(X_ensemble_test_lr)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6373735935051987 Parameter: {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}")


        st.subheader("2.7.3 Meta-Classifier Support Vector Machine (SVM)")
        st.write("Untuk membuat SVM sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_svm dari dataset training
        ensemble_train_svm = results_validation.drop('y_pred_svm', axis=1)

        # menghapus y_pred_svm dari dataset testing
        ensemble_test_svm = results_testing.drop('y_pred_svm', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_svm, y_ensemble_train_svm = ensemble_train_svm.drop(['y_actual', 'test_index'], axis=1), ensemble_train_svm['y_actual']
        X_ensemble_test_svm, y_ensemble_test_svm = ensemble_test_svm.drop('y_actual', axis=1), ensemble_test_svm['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_ensemble_svm = GridSearchCV(svm, parameters_svm, cv=kf, scoring='accuracy')
        grid_search_ensemble_svm.fit(X_ensemble_train_svm, y_ensemble_train_svm)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_svm.best_score_)
        print(grid_search_ensemble_svm.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_svm = grid_search_ensemble_svm.best_estimator_
        y_pred_ensemble_svm = best_model_ensemble_svm.predict(X_ensemble_test_svm)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6508617006124484 Parameter: {'C': 1, 'gamma': 0.001, 'kernel': 'linear'}")

        st.subheader("2.7.4 Meta-Classifier Artificial Neural Network (ANN)")
        st.write("Untuk membuat ANN sebagai meta-classifier langkah awalnya adalah menghapus y_pred_lr dari dataset yang dikumpulkan baik untuk training dan juga testing")

        code = '''
        # menghapus y_pred_ann dari dataset training
        ensemble_train_ann = results_validation.drop('y_pred_ann', axis=1)

        # menghapus y_pred_ann dari dataset testing
        ensemble_test_ann = results_testing.drop('y_pred_ann', axis=1)

        # memisahkan antara fitur dengan target
        X_ensemble_train_mlp, y_ensemble_train_mlp = ensemble_train_mlp.drop(['y_actual', 'test_index'], axis=1), ensemble_train_mlp['y_actual']
        X_ensemble_test_mlp, y_ensemble_test_mlp = ensemble_test_mlp.drop('y_actual', axis=1), ensemble_test_mlp['y_actual']
        '''

        st.code(code, language="python")

        st.write("Selanjutnya tinggal melakukan training untuk meta-classifier knn dengan code program dibawah ini.")

        code = '''
        # menggunakan model knn pada sklearn
        svm = SVC()

        # menentukan parameter yang akan di tuning
        parameters_svm = {
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': [0.001, 0.01, 0.1, 1],
            'C' : [0.01, 1, 10, 100, 1000]
        }

        # melakukan grid search pada svm
        grid_search_ensemble_mlp = GridSearchCV(mlp, parameters_mlp, cv=kf, scoring='accuracy')
        grid_search_ensemble_mlp.fit(X_ensemble_train_mlp, y_ensemble_train_mlp)

        # menampilkan hasil parameter terbaik
        print(grid_search_ensemble_mlp.best_score_)
        print(grid_search_ensemble_mlp.best_params_)

        # melakukan prediksi pada data testing menggunakan parameter terbaik
        best_model_ensemble_mlp = grid_search_ensemble_mlp.best_estimator_
        y_pred_ensemble_mlp = best_model_ensemble_mlp.predict(X_ensemble_test_mlp)
        '''

        st.code(code, language="python")

        st.write("Output:")
        st.text("Hasil Terbaik: 0.6542942600769122 Parameter: {'activation': 'tanh', 'hidden_layer_sizes': (4,), 'learning_rate_init': 0.001, 'max_iter': 500}")

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
