import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu


# Judul web
st.title('Aplikasi Prediksi Lama Rawat Inap Pasien Demam Berdarah Dalam Kategori Periode Hari')

#navigasi sidebar
# horizontal menu
selected2 = option_menu(None, ["Data", "Preprocessing", "Implementasi"], 
    icons=['house', 'filter', 'gear'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

#halaman Data
if (selected2 == 'Data') :
    st.subheader('Deskripsi Data')

    st.write("Dataset yang digunakan terdiri dari 6 kolom utama yang berfungsi sebagai fitur untuk memprediksi lama rawat inap (Length of Stay/LOS) pasien demam berdarah.")
    data = pd.read_csv('https://raw.githubusercontent.com/risma260/RISET/refs/heads/main/dataset.csv', sep=';')
    data_without_no = data.drop(columns=['NO'])
    data_without_no.insert(0, 'No.', range(1, len(data_without_no) + 1))
    st.write(data_without_no)

if (selected2 == 'Preprocessing') :
    st.subheader('Preprocessing Data')

    st.write("Data di proses dahulu sebelum dimasukkan ke model, yaitu mengubah nilai kategori menjadi nilai numerik dan mengisi nilai yang hilang menggunakan KNN Imputation.")
    data2 = pd.read_csv('https://raw.githubusercontent.com/risma260/RISET/refs/heads/main/dataset_preprocessing.csv', sep=',')
    data2.insert(0, 'No.', range(1, len(data2) + 1))
    st.write(data2)

         
# Halaman Implementasi
if selected2 == 'Implementasi':
    st.subheader('Implementasi')

    # Membaca model
    dbd_model = pickle.load(open('model_xgboost.pkl', 'rb'))


    # Membagi kolom untuk input
    col1, col2 = st.columns(2)

    with col1:
        umur = st.number_input('Umur (tahun)', min_value=0)
        trombosit = st.number_input('Jumlah Trombosit (x10^3/μL)', min_value=0)
        hct = st.number_input('Hematokrit (HCT %)', min_value=0.0, step=0.1)

    with col2:
        hb = st.number_input('Hemoglobin (HB g/dL)', min_value=0.0, step=0.1)
        jenis_kelamin = st.selectbox('Jenis Kelamin', ['Laki-laki', 'Perempuan'])
        diagnosis = st.selectbox('Jenis Demam', ['DD', 'DBD', 'DSS'])

    # Encoding gender dan diagnosis
    jenis_kelamin_mapping = {'Laki-laki': 0, 'Perempuan': 1}
    diagnosis_mapping = {'DD': 0, 'DBD': 1, 'DSS': 2}

    jenis_kelamin_encoded = jenis_kelamin_mapping[jenis_kelamin]
    diagnosis_encoded = diagnosis_mapping[diagnosis]

    # Membuat tombol untuk prediksi
    if st.button('Prediksi Lama Rawat Inap'):
        # Prediksi dengan model
        prediksi_lama_rawat = dbd_model.predict([[umur, trombosit, hct, hb, jenis_kelamin_encoded, diagnosis_encoded]])

        # Menampilkan hasil prediksi
        st.subheader('Hasil Prediksi')
        st.write(f"Perkiraan lama rawat inap: {round(prediksi_lama_rawat[0])} hari")
        