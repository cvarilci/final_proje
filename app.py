# --- 1. Gerekli Kütüphaneleri Yükleme ---
import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

# --- 2. FastAPI Uygulamasını ve Template Dizinini Tanımlama ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- 3. Kaydedilmiş Modeli ve Scaler'ı Yükleme ---
# Bu dosyalar, app.py ile aynı klasörde olmalı.
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# --- 4. Modelin Eğitildiği Sütunları Tanımlama (ÇOK ÖNEMLİ) ---
# Modelinize veri göndermeden önce, kullanıcıdan gelen verinin
# modelin eğitildiği formatla (aynı sütunlar ve aynı sıra) eşleşmesi gerekir.
# Bu listeyi, eğitim script'inizdeki X.columns'tan alabilirsiniz.
model_columns = [
    'age', 'bmi', 'children', 'sex_male', 'smoker_yes',
    'region_northwest', 'region_southeast', 'region_southwest'
]

# --- 5. Ana Sayfa (GET İsteği) ---
# Kullanıcı web sitesini ilk ziyaret ettiğinde bu fonksiyon çalışır.
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Sadece index.html'i render et ve kullanıcıya göster.
    return templates.TemplateResponse("index.html", {"request": request})

# --- 6. Tahmin Fonksiyonu (POST İsteği) ---
# Kullanıcı formdaki "Tahmin Et" butonuna tıkladığında bu fonksiyon çalışır.
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    sex: str = Form(...),
    bmi: float = Form(...),
    children: int = Form(...),
    smoker: str = Form(...),
    region: str = Form(...)
):
    # Adım 1: Kullanıcıdan gelen veriyi bir dictionary'ye koy.
    data = {
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }

    # Adım 2: Bu dictionary'den tek satırlık bir Pandas DataFrame oluştur.
    input_df = pd.DataFrame([data])

    # Adım 3: Kategorik verileri one-hot encoding'e çevir (get_dummies).
    input_df = pd.get_dummies(input_df)

    # Adım 4: Gelen veriyi, modelin eğitildiği sütun yapısına zorla.
    # .reindex() metodu, eksik sütunları ekler (değerini 0 yapar) ve fazlalıkları atar.
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Adım 5: Sayısal sütunları, daha önce kaydettiğimiz scaler ile ölçeklendir.
    # DİKKAT: Burada .fit_transform() DEĞİL, sadece .transform() kullanılır!
    numerical_cols = ['age', 'bmi', 'children']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Adım 6: Hazırlanmış veriyle modelden tahmini al.
    prediction = model.predict(input_df)

    # Adım 7: Tahmin sonucunu formatla (virgülden sonra 2 basamak).
    formatted_prediction = f"{prediction[0]:,.2f}"

    # Adım 8: Sonucu göstermek için index.html'i tekrar render et ve tahmini sayfaya gönder.
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": formatted_prediction
    })

# Bu dosya doğrudan çalıştırıldığında uvicorn sunucusunu başlatmak için (opsiyonel)
if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)

    