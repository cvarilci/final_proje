# Gerekli kütüphaneleri içeri aktaralım
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükleyelim
# 'insurance.csv' dosyasının kod dosyanızla aynı klasörde olduğundan emin olun
df = pd.read_csv('insurance.csv')

# --- Veriyi Tanıyalım ---

# 1. Veri setinin ilk 5 satırını görelim
print("Veri Setinin İlk 5 Satırı:")
print(df.head())
print("\n" + "="*80 + "\n")

# 2. Veri setinin yapısı hakkında bilgi alalım (sütun adları, veri tipleri, boş değerler)
print("Veri Seti Bilgileri:")
df.info()
print("\n" + "="*80 + "\n")
# Gördüğümüz gibi, veri setinde hiç boş (null) değer yok. Bu harika bir durum!

# 3. Sayısal sütunlar için temel istatistikleri görelim (ortalama, standart sapma vb.)
print("Sayısal Veri İstatistikleri:")
print(df.describe())
print("\n" + "="*80 + "\n")


# --- Veriyi Görselleştirelim ---

# Seaborn'un stilini ayarlayalım
sns.set_style('whitegrid')

# 1. Masrafların (charges) dağılımını görelim
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, bins=40)
plt.title('Tıbbi Masrafların Dağılımı', fontsize=15)
plt.xlabel('Masraflar (Charges)', fontsize=12)
plt.ylabel('Frekans', fontsize=12)
plt.show()
# Yorum: Masrafların büyük bir çoğunluğu 15.000$'ın altında toplanmış. Dağılım sağa çarpık.

# 2. Sigara içenler ve içmeyenlerin masraflarını karşılaştıralım
plt.figure(figsize=(8, 6))
sns.boxplot(x='smoker', y='charges', data=df)
plt.title('Sigara İçme Durumuna Göre Tıbbi Masraflar', fontsize=15)
plt.xlabel('Sigara İçiyor mu?', fontsize=12)
plt.ylabel('Masraflar (Charges)', fontsize=12)
plt.show()
# Yorum: Sigara içenlerin tıbbi masrafları, içmeyenlere göre çok daha yüksek. Bu çok önemli bir özellik!

# 3. Yaş ve masraflar arasındaki ilişkiyi görelim
plt.figure(figsize=(10, 6))
sns.scatterplot(x='age', y='charges', data=df, hue='smoker', alpha=0.7)
plt.title('Yaş ve Tıbbi Masraflar Arasındaki İlişki', fontsize=15)
plt.xlabel('Yaş', fontsize=12)
plt.ylabel('Masraflar (Charges)', fontsize=12)
plt.show()
# Yorum: Yaş arttıkça masraflar da artma eğiliminde. Ayrıca sigara içenlerde bu artış çok daha belirgin.

# 4. BMI (Vücut Kitle İndeksi) ve masraflar arasındaki ilişki
plt.figure(figsize=(10, 6))
sns.scatterplot(x='bmi', y='charges', data=df, hue='smoker', alpha=0.7)
plt.title('BMI ve Tıbbi Masraflar Arasındaki İlişki', fontsize=15)
plt.xlabel('BMI', fontsize=12)
plt.ylabel('Masraflar (Charges)', fontsize=12)
plt.show()
# Yorum: BMI arttıkça masrafların da arttığı söylenebilir, ancak bu ilişki yaş kadar güçlü görünmüyor. 
# Özellikle BMI'si yüksek olan sigara içicilerde masraflar çok daha fazla.
# Genel Değerlendirme: Sigara içme durumu, tıbbi masraflar üzerinde en belirgin etkiye sahip özellik gibi görünüyor.
# Yaş ve BMI de masraflarla pozitif korelasyona sahip, ancak sigara içme durumu kadar güçlü değil.
# Bu görselleştirmeler, modelleme aşamasında hangi özelliklerin daha önemli olabileceği konusunda bize fikir veriyor.

# --- Korelasyon Analizi İçin Veriyi Hazırlama ---

# Korelasyon matrisi sadece sayısal değerlerle çalışır.
# Bu yüzden kategorik sütunları (sex, smoker, region) sayısala çevireceğiz.
# Bu işleme "One-Hot Encoding" denir ve bir sonraki adımda kalıcı olarak yapacağız.
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# get_dummies ne yaptı bir görelim:
# - sex -> sex_male (1 ise erkek, 0 ise kadın)
# - smoker -> smoker_yes (1 ise sigara içiyor, 0 ise içmiyor)
# - region -> region_northwest, region_southeast, region_southwest (1 ise o bölgede, 0 ise değil)
# 'drop_first=True' parametresi gereksiz bir sütun oluşmasını engeller. 
# Örneğin sex_male=0 ise zaten kadın olduğunu anlarız, ikinci bir sütuna gerek kalmaz.

print("Dönüştürülmüş Verinin İlk 5 Satırı:")
print(df_encoded.head())
print("\n" + "="*50 + "\n")

# --- Korelasyon Matrisini Oluşturma ve Görselleştirme ---

# Şimdi tüm veriler sayısal olduğu için korelasyon matrisini hesaplayabiliriz.
correlation_matrix = df_encoded.corr()

# Isı haritasını (heatmap) çizelim
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
# annot=True: Değerleri karelerin üzerine yazar.
# cmap='coolwarm': Renk paletini belirler (pozitif için sıcak, negatif için soğuk renkler).
# fmt='.2f': Değerleri virgülden sonra 2 basamakla formatlar.

plt.title('Değişkenlerin Korelasyon Isı Haritası', fontsize=16)
plt.show()

# --- 1. Gerekli Kütüphaneler ve Veri Hazırlığı (Önceki Adımlardan) ---
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

# Modeller
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Veriyi yükle, encode et, ayır
df = pd.read_csv('insurance.csv')
df_encoded = pd.get_dummies(df, drop_first=True)
X = df_encoded.drop('charges', axis=1)
y = df_encoded['charges']

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Veriyi ölçeklendir
numerical_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Sonuçları saklamak için bir sözlük oluşturalım
results = {}

# --- 2. Model 1: SVR (Destek Vektör Regresyonu) ---
print("--- Model 1: SVR Tuning Başlıyor ---")
# Bu sefer SADECE rbf kernel'i zorlayacağız.
svr_params = {
    'C': [1000, 5000, 10000],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}
svr_grid = GridSearchCV(SVR(), svr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
svr_grid.fit(X_train, y_train)
best_svr = svr_grid.best_estimator_
y_pred_svr = best_svr.predict(X_test)

results['SVR'] = {
    'Best Params': svr_grid.best_params_,
    'R2 Score': r2_score(y_test, y_pred_svr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_svr))
}
print("--- SVR Tuning Tamamlandı ---\n")


# --- 3. Model 2: Random Forest Regressor ---
print("--- Model 2: Random Forest Tuning Başlıyor ---")
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
y_pred_rf = best_rf.predict(X_test)

results['Random Forest'] = {
    'Best Params': rf_grid.best_params_,
    'R2 Score': r2_score(y_test, y_pred_rf),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf))
}
print("--- Random Forest Tuning Tamamlandı ---\n")


# --- 4. Model 3: Gradient Boosting Regressor ---
print("--- Model 3: Gradient Boosting Tuning Başlıyor ---")
gbr_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 4]
}
gbr_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gbr_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
gbr_grid.fit(X_train, y_train)
best_gbr = gbr_grid.best_estimator_
y_pred_gbr = best_gbr.predict(X_test)

results['Gradient Boosting'] = {
    'Best Params': gbr_grid.best_params_,
    'R2 Score': r2_score(y_test, y_pred_gbr),
    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_gbr))
}
print("--- Gradient Boosting Tuning Tamamlandı ---\n")


# --- 5. Final Sonuçlarının Karşılaştırılması ---
print("="*50)
print("MODEL PERFORMANS KARŞILAŞTIRMASI")
print("="*50)

for model_name, metrics in results.items():
    print(f"Model: {model_name}")
    print(f"  - En İyi Parametreler: {metrics['Best Params']}")
    print(f"  - R-Kare Skoru: {metrics['R2 Score']:.4f}")
    print(f"  - Kök Ortalama Kare Hata (RMSE): ${metrics['RMSE']:,.2f}")
    print("-" * 30)

import pickle

# Kazanan modelimiz Gradient Boosting idi. 
# GridSearchCV sonucunda bulunan en iyi modeli bir değişkene atayalım.
champion_model = gbr_grid.best_estimator_

# --- Modeli Pickle ile Kaydetme ---
# Dosyayı 'wb' (write binary) modunda açıyoruz ve pickle.dump ile nesneyi içine yazıyoruz.
with open('model.pkl', 'wb') as file:
    pickle.dump(champion_model, file)

# --- Scaler'ı Pickle ile Kaydetme ---
# Scaler'ı da aynı şekilde kaydediyoruz. Bu çok önemli!
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("\n" + "="*50)
print("Şampiyon model 'model.pkl' olarak kaydedildi.")
print("Veri ölçekleyici 'scaler.pkl' olarak kaydedildi.")
print("Artık web uygulaması için hazırız!")
print("="*50)


