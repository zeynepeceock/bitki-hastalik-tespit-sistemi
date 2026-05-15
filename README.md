# Bitki Hastalık Tespit Sistemi

Bu proje, derin öğrenme kullanılarak bitki yapraklarındaki hastalıkların tespit edilmesi amacıyla geliştirilmiştir.

## Kullanılan Teknolojiler

- Python
- TensorFlow / Keras
- Flask
- MobileNetV2
- ResNet50
- HTML / CSS

## Veri Setleri

- PlantVillage Dataset
- PlantDoc Dataset

## Kullanılan Modeller

| Model | Validation Accuracy |
|---|---|
| MobileNetV2 | %91.99 |
| ResNet50 | %48.45 |

Bu çalışmada en başarılı sonuç MobileNetV2 modeli ile elde edilmiştir.

## Proje Özellikleri

- Yaprak görseli yükleme
- Hastalık tespiti
- Bitki türü belirleme
- Güven oranı gösterimi
- Web tabanlı arayüz

## Projeyi Çalıştırma

### Gerekli Kütüphaneler

```bash
pip install -r requirements.txt
```

### Uygulamayı Başlatma

```bash
python app.py
```

### Tarayıcıda Açma

```text
http://127.0.0.1:5000
```

## Geliştirici

ECE ZEYNEP OCAK