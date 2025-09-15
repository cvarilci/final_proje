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
print("\n" + "="*50 + "\n")

# 2. Veri setinin yapısı hakkında bilgi alalım (sütun adları, veri tipleri, boş değerler)
print("Veri Seti Bilgileri:")
df.info()
print("\n" + "="*50 + "\n")
# Gördüğümüz gibi, veri setinde hiç boş (null) değer yok. Bu harika bir durum!

# 3. Sayısal sütunlar için temel istatistikleri görelim (ortalama, standart sapma vb.)
print("Sayısal Veri İstatistikleri:")
print(df.describe())
print("\n" + "="*50 + "\n")


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