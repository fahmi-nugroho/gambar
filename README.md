# Laporan Proyek Machine Learning - Fahmi Nugroho Alibasyah

## Domain Proyek
Untuk proyek pertama ini saya memilih klasifikasi kanker payudara sebagai masalah yang akan saya selesaikan apakah termasuk ke jenis benign atau malignnant. Saya memlih masalah tersebut karena kanker payudara merupakan jenis kanker yang paling sering terjadi, hal ini karena selama setahun kanker payudara mempengaruhi lebih dari dua juta Wanita. Dengan adanya sebuah program untuk mengklasifikasikan kanker payudara termasuk ke jenis benign atau malignant maka akan memudahkan dan mempercepat penanganan sesuai jenis kanker yang dihadapi pasien.

Referensi :
* S. Sharma, A. Aggarwal and T. Choudhury, "Breast Cancer Detection Using Machine Learning Algorithms," 2018 International Conference on Computational Techniques, Electronics and Mechanical Systems (CTEMS), 2018, pp. 114-118, doi: 10.1109/CTEMS.2018.8769187.

## Business Understanding
* Pernyataan masalah
    * Bagaimana menentukan jenis kanker payudara yang sedang diderita pasien.
* Tujuan
    * Membuat model machine learning yang dapat mengklasifikasi jenis kanker yang sedang diderita pasien.
* Pernyataan solusi
    * Menggunakan model machine learning dengan pendekatan K-Nearest Neighbor.
    * Menggunakan model machine learning dengan pendekatan Random Forest.

## Data Understanding
Data yang saya gunakan merupakan data Breast Cancer Wisconsin (Diagnostic) Data Set yang saya ambil pada website [Kaggle](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data). Data tersebut berisi beberapa fitur seperti :
* Diagnosis (M = malignant, B = benign)
* Radius (rata-rata jarak dari tengah sampai titik pada perimeter)
* Texture (standar deviasi dari nilai gray-scale)
* Perimeter
* Area
* Smoothness (variasi lokal didalam panjang radius)
* Compactness (perimeter^2 / area - 1.0)
* Concavity (keparahan bagian cekung dari kontur)
* Concave points (jumlah bagian cekung dari kontur)
* Symmetry
* Fractal dimension ("coastline approximation" - 1)

Masing-masing fitur diatas (kecuali diagnosis) memliki 3 jenis, yaitu rata-rata, standar eror, dan rata-rata terbesar (worst). Contohnya radius_mean, radius_se, dan radius_worst.

Data diagnosis

![Visualisasi Diagnosis](https://github.com/fahmi-nugroho/gambar/blob/main/gambar4.png)

Data untuk rata-rata (mean)

![Visualisasi Dataset Rata-Rata](https://github.com/fahmi-nugroho/gambar/blob/main/gambar1.png)

Data untuk standar eror (se)

![Visualisasi Dataset Standar Eror](https://github.com/fahmi-nugroho/gambar/blob/main/gambar2.png)

Data untuk rata-rata terbesar (worst)

![Visualisasi Dataset Rata-Rata Terbesar](https://github.com/fahmi-nugroho/gambar/blob/main/gambar3.png)

## Data Preparation
* Melakukan encoding untuk fitur diagnosis
Encoding dilakukan untuk mengubah fitur yang berupa kategori menjadi berupa string. Karena model machine learning tidak bisa memproses data dengan tipe string.
    ```python
    data = pd.concat([data, pd.get_dummies(data['diagnosis'], prefix='diagnosis', drop_first=True)],axis=1)
    ```
* Melakukan standarisasi data uji dan data validasi
Standarisasi data dilakukan karena algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses scaling dan standarisasi membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
    ```python
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train = scaler_train.transform(X_train)
    
    scaler_val = StandardScaler()
    scaler_val.fit(X_test)
    X_test = scaler_val.transform(X_test)
    ```
* Melakukan pembagian dataset untuk data latih dan data validasi
Dengan melakukan pembagian dataset kita dapat menilai bagaimana performa yang dihasilkan model kita ketika bertemu data-data yang belum pernah dilihat pada proses latihan sebelumnya.
    ```python
    X = data.drop(["diagnosis_M"],axis =1)
    y = data["diagnosis_M"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    ```

# Modeling
Saya menggunakan dua pendekatan dalam membuat model machine learning untuk menyelesaikan permasalahan klasifikasi jenis kanker payudara.
* K-Nearest Neighbor
    ```python
    knn = KNeighborsClassifier(n_neighbors = 2)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    ```
* Random Forest
    ```python
    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)
    y_pred_RF=RF.predict(X_test)
    ```

## Evaluation
Untuk menilai performa model saya menggunakan tiga metrik evaluasi. Saya memilih ketiga metrik dibawah karena masalah yang saya selesaikan merupakan masalah klasifikasi.
* Confussion Matrix
Confusion matrix juga sering disebut error matrix. Pada dasarnya confusion matrix memberikan informasi perbandingan hasil klasifikasi yang dilakukan oleh sistem (model) dengan hasil klasifikasi sebenarnya. Confusion matrix berbentuk tabel matriks yang menggambarkan kinerja model klasifikasi pada serangkaian data uji yang nilai sebenarnya diketahui.

![Tabel Confussion Matrix](https://github.com/fahmi-nugroho/gambar/blob/main/confussionmatrix.png)
* Classfication Report
    * Precission
Precission adalah kemampuan pengklasifikasi untuk tidak melabeli instance positif yang sebenarnya negatif. Untuk setiap kelas, itu didefinisikan sebagai rasio positif benar dengan jumlah positif benar dan positif palsu. ![Precission](https://github.com/fahmi-nugroho/gambar/blob/main/precision.png)
    * Recall
Recall adalah kemampuan classifier untuk menemukan semua instance positif. Untuk setiap kelas itu didefinisikan sebagai rasio positif benar dengan jumlah positif benar dan negatif palsu. ![Precission](https://github.com/fahmi-nugroho/gambar/blob/main/recall.png)
    * F1 Score
F1 Score adalah rata-rata harmonik tertimbang dari presisi dan daya ingat sehingga skor terbaik adalah 1,0 dan yang terburuk adalah 0,0. F1 Score lebih rendah dari ukuran akurasi karena mereka menanamkan presisi dan mengingat ke dalam perhitungan mereka. Sebagai aturan praktis, rata-rata tertimbang F1 harus digunakan untuk membandingkan model pengklasifikasi, bukan akurasi global. ![Precission](https://github.com/fahmi-nugroho/gambar/blob/main/f1.png)
    * Support
Support adalah jumlah kemunculan aktual kelas dalam kumpulan data yang ditentukan. Support yang tidak seimbang dalam data pelatihan dapat menunjukkan kelemahan struktural dalam skor pengklasifikasi yang dilaporkan dan dapat menunjukkan perlunya pengambilan sampel bertingkat atau penyeimbangan kembali. Support tidak berubah antar model melainkan mendiagnosis proses evaluasi.
* Accuracy Score

![Accuracy](https://github.com/fahmi-nugroho/gambar/blob/main/accuracy.png)

Kode yang saya gunakan untuk mengevaluasi model:

    ```python
    print('==================== K-Nearest Neighbor ====================')
    
    print('=> Confusion Matrix')
    print(confusion_matrix(y_test,y_pred_knn))
    print('=> Classification Report')
    print(classification_report(y_test,y_pred_knn))
    print('=> Accuracy Score')
    print(accuracy_score(y_test, y_pred_knn))
    
    print('\n\n\n==================== Random Forest ====================')
    print('=> Confusion Matrix')
    print(confusion_matrix(y_test,y_pred_RF))
    print('=> Classification Report')
    print(classification_report(y_test,y_pred_RF))
    print('=> Accuracy Score')
    print(accuracy_score(y_test, y_pred_RF))
    ```

Hasil Evaluasi K-Nearest Neighbor

![Evaluasi KNN](https://github.com/fahmi-nugroho/gambar/blob/main/gambar5.png)

Hasil Evaluasi Random Forest

![Evaluasi RF](https://github.com/fahmi-nugroho/gambar/blob/main/gambar6.png)
