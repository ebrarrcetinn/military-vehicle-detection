# Askeri Araç Tespit Sistemi

Bu proje, kamera veya video akışından gerçek zamanlı askeri araç tespiti yapan bir C++ uygulamasıdır. NVIDIA Jetson platformunda çalışmak üzere tasarlanmış olup Windows ortamında geliştirilmiş ve test edilmiştir. Sistem her video karesinde tank, zırhlı araç, asker, top ve askeri kamyonu tespit eder, tespit edilen her nesnenin etrafına yeşil kutu çizer ve güven yüzdesini ekranda gösterir.

---

## Proje Hakkında

Savunma sanayii uygulamalarında gerçek zamanlı görüntü işleme kritik bir gereksinimdir. Bu sistem o ihtiyacı karşılamak amacıyla geliştirildi. Geliştirme süreci iki aşamadan oluştu: önce Python ile model eğitimi yapıldı, ardından model ONNX formatına export edilerek tamamen C++ tabanlı bir inference pipeline kuruldu. Bu yaklaşım savunma sanayiinde yaygın kullanılan bir yöntemdir; araştırma ekibi Python ile model geliştirir, yazılım mühendisleri C++ ile sisteme entegre eder.

---

## Kullanılan Teknolojiler

**C++17** savunma ve gömülü sistemlerde endüstri standardı programlama dilidir. Python'a kıyasla çok daha hızlı çalışır ve bellek yönetimi öngörülebilirdir. NVIDIA Jetson gibi kaynak kısıtlı edge cihazlarda gerçek zamanlı işlem yapabilmek için C++ tercih edildi. Aynı kaynak kod platformdan bağımsız olarak hem Windows hem Linux üzerinde derlenebilir.

**OpenCV** görüntü işlemenin endüstri standardı kütüphanesidir. Kamera ve video okuma, görüntü boyutlandırma, piksel dönüşümleri ve ekrana çizim işlemleri bu kütüphane ile yapıldı. OpenCV içindeki dnn modülü eğitilmiş yapay zeka modellerini Python olmadan doğrudan C++ tarafında çalıştırmayı sağlar. ONNX formatındaki modelleri yükleyebilir ve GPU hızlandırması için CUDA backend seçilebilir.

**YOLOv8n** gerçek zamanlı nesne tespiti için en yaygın kullanılan derin öğrenme mimarisidir. Görüntüyü tek bir geçişte işler ve aynı anda hem nesnenin konumunu hem de sınıfını tahmin eder. Kullandığımız nano versiyonu 3 milyon parametre içerir ve en küçük ile en hızlı YOLO versiyonudur. NVIDIA Jetson gibi edge cihazlarda yüksek FPS değerlerine ulaşmak için nano versiyonu seçildi.

**ONNX** evrensel bir model formatıdır. Model eğitimi Python ve PyTorch ile yapıldı çünkü bu ekosistem model geliştirme için en olgun araçlara sahip. Ancak üretim ortamında sistem C++ ile çalışıyor. ONNX bu iki dünyanın köprüsüdür. Python, C++, Java, C# dahil pek çok dil ve platform tarafından okunabilir.

**CMake** cross-platform derleme sistemidir. Aynı CMakeLists.txt dosyası hem Windows hem Linux üzerinde çalışır. Bu sayede geliştirme Windows'ta yapılıp üretim Jetson üzerinde kolayca gerçekleştirilebilir.

---

## Tespit Edilen Sınıflar

| Sınıf | Açıklama |
|---|---|
| tank | Tank |
| armoured-vehicle | Zırhlı araç |
| artillery | Top, obüs |
| person | Asker |
| truck | Askeri kamyon |

---

## Sistem Nasıl Çalışıyor

Kamera veya video dosyasından her kare okunur. Görüntü 640x640 boyutuna getirilir. Piksel değerleri 0-255 arasından 0-1 arasına normalize edilir çünkü model eğitilirken bu format kullanıldı. OpenCV görüntüleri BGR formatında okur ancak YOLOv8 RGB formatında eğitildi, bu yüzden dönüşüm yapılır. YOLOv8n modeli bu girdi üzerinde inference yapar ve 8400 olası tespit üretir. Her tespit için güven skoru hesaplanır, 0.5 altındakiler atılır. NMS ile üst üste gelen kutular temizlenir ve kalan kutular görüntü üzerine çizilir.

NMS aynı nesne için üretilen birden fazla kutuyu temizler. Model bir tank için 20 üst üste kutu üretebilir. NMS kutular arasındaki örtüşme oranını hesaplar ve en yüksek güvenli kutuyu seçerek diğerlerini siler. IoU eşiği 0.4 olarak ayarlandı.

Güven eşiği 0.5 olarak ayarlandı. Sahaya göre değiştirilebilir. Keşif görevinde düşürülerek hiçbir hedefin kaçırılmaması sağlanır. Hedefleme sisteminde yükseltilerek yanlış alarm riski azaltılır.

---

## Model Performansı

| Metrik | Değer |
|---|---|
| Dataset | Military Vehicle Dataset, Roboflow Universe |
| Görüntü Sayısı | 10.218 |
| Sınıf Sayısı | 5 |
| Epoch | 15 |
| mAP50 | 0.535 |
| mAP50-95 | 0.321 |
| Tank mAP50 | 0.645 |
| Inference CPU | 43ms/kare |

---

## Model Eğitimi

Transfer learning yöntemi kullanıldı. YOLOv8n modeli ImageNet üzerinde milyonlarca görüntü ile önceden eğitilmiş genel bir modeldir. Bu model alınarak military vehicle dataseti üzerinde fine-tune yapıldı. Transfer learning sayesinde sıfırdan eğitim yerine çok daha az veri ve süre ile yüksek performans elde edildi.

---

## Kurulum

### Gereksinimler
- CMake 3.10+
- OpenCV 4.x
- C++17 destekli derleyici

### Derleme
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Çalıştırma
```bash
cd Release
./tanktespiti.exe
```

---

## Jetson Entegrasyonu

main.cpp içinde sadece 2 satır değişiyor:

```cpp
// Geliştirme - Windows CPU
model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

// Üretim - Jetson GPU
model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

Linux ortamında derleme:
```bash
sudo apt install cmake libopencv-dev
mkdir build && cd build
cmake ..
make
./tanktespiti
```

---

# Military Vehicle Detection System

This project is a real-time military vehicle detection application written in C++, designed to run on NVIDIA Jetson platforms. It was developed and tested on Windows. The system detects tanks, armoured vehicles, soldiers, artillery and military trucks in every video frame, draws bounding boxes around detected objects and displays confidence percentages on screen.

---

## About

Real-time image processing is a critical requirement in defence industry applications. This system was developed to meet that need. The development process consisted of two stages: first, model training was done in Python, then the model was exported to ONNX format and a fully C++ based inference pipeline was built. This approach is widely used in the defence industry.

---

## Technologies Used

**C++17** is the industry standard programming language for defence and embedded systems. It is significantly faster than Python and provides predictable memory management. C++ was chosen to achieve real-time processing on resource-constrained edge devices like NVIDIA Jetson. The same source code compiles on both Windows and Linux.

**OpenCV** is the industry standard library for computer vision. Camera and video reading, image resizing, pixel transformations and drawing operations were all done with this library. The dnn module inside OpenCV allows running trained AI models directly in C++ without Python. It supports CUDA backend for GPU acceleration.

**YOLOv8n** is the most widely used deep learning architecture for real-time object detection. It processes the image in a single pass and simultaneously predicts both location and class. The nano version contains 3 million parameters and is the smallest and fastest YOLO version, ideal for edge devices.

**ONNX** is a universal model format. The model was trained in Python and PyTorch, then exported to ONNX to run in C++. It can be read by Python, C++, Java, C# and many other languages.

**CMake** is a cross-platform build system. The same CMakeLists.txt file works on both Windows and Linux.

---

## Detected Classes

| Class | Description |
|---|---|
| tank | Tank |
| armoured-vehicle | Armoured vehicle |
| artillery | Artillery, howitzer |
| person | Soldier |
| truck | Military truck |

---

## How It Works

Each frame is read from a camera or video file. The image is resized to 640x640. Pixel values are normalized from 0-255 to 0-1. OpenCV reads images in BGR format but YOLOv8 was trained on RGB, so a conversion is performed. The model produces 8400 possible detections. Those below 0.5 confidence are discarded. NMS removes overlapping boxes and remaining boxes are drawn on screen.

NMS removes duplicate boxes for the same object. The model can produce 20 overlapping boxes for a single tank. NMS keeps the highest confidence box and removes the rest. IoU threshold is set to 0.4.

The confidence threshold is 0.5. It can be adjusted per mission. For reconnaissance it can be lowered. For targeting systems it can be raised to reduce false alarms.

---

## Model Performance

| Metric | Value |
|---|---|
| Dataset | Military Vehicle Dataset, Roboflow Universe |
| Images | 10,218 |
| Classes | 5 |
| Epochs | 15 |
| mAP50 | 0.535 |
| mAP50-95 | 0.321 |
| Tank mAP50 | 0.645 |
| CPU Inference | 43ms/frame |

---

## Model Training

Transfer learning was used. YOLOv8n is pre-trained on millions of ImageNet images. It was fine-tuned on the military vehicle dataset. Thanks to transfer learning, high performance was achieved with much less data and time compared to training from scratch.

---

## Installation

### Requirements
- CMake 3.10+
- OpenCV 4.x
- C++17 compatible compiler

### Build
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

### Run
```bash
cd Release
./tanktespiti.exe
```

---

## Jetson Deployment

Only 2 lines change in main.cpp:

```cpp
// Development - Windows CPU
model.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
model.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

// Production - Jetson GPU
model.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
model.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
```

Linux build:
```bash
sudo apt install cmake libopencv-dev
mkdir build && cd build
cmake ..
make
./tanktespiti
```

---

