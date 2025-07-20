# 👁️‍🗨️ Attention Monitoring System

A real-time head pose detection and attention monitoring system that tracks whether a person is focused or distracted using webcam input. Perfect for use in classrooms, exam proctoring, or workplace environments.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Facial_Landmarks-orange?logo=google)
![Pandas](https://img.shields.io/badge/Pandas-Data_Logging-informational?logo=pandas)
![Platform](https://img.shields.io/badge/Platform-Windows%7CLinux%7CMac-lightgrey)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

---

## 📌 Features

- ✅ **Real-Time Face & Head Pose Detection**  
  Detects the face and estimates head pose angles using MediaPipe and OpenCV.

- ✅ **Distraction Detection & Alerts**  
  Monitors head angles and notifies if the user is distracted.

- ✅ **Auto Image Capture**  
  Captures and stores images when distraction is detected.

- ✅ **Attention Logging System**  
  Logs timestamped data (pitch, yaw, roll, attention status) into a CSV.

- ✅ **Modular, Beginner-Friendly Codebase**  
  Clean, readable, and well-commented Python code for learning and extension.

---

## 🧰 Technologies Used

| Technology | Description |
|------------|-------------|
| ![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) | Programming Language |
| ![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green?logo=opencv) | For video processing and drawing |
| ![MediaPipe](https://img.shields.io/badge/MediaPipe-Facial_Landmarks-orange?logo=google) | For facial and landmark detection |
| ![Pandas](https://img.shields.io/badge/Pandas-Data_Logging-informational?logo=pandas) | For logging attention data |

---

## 🚀 Installation

1. **Clone the Repository**
```bash
git clone https://github.com/vamsiprasad-ctrl/attention_Monioring_system.git
cd attention_Monioring_system
```
---

2. **🔧 Install Dependencies**
   ```bash
   pip install opencv-python mediapipe pandas
   ```
   ---
3. **🔧  How to Run**
   ```bash
   python attention_monitor.py or py attention_monitor.py
   ```
---
**📂 Output**

- `captured/` — Stores snapshots of the user when distraction is detected.
- `attention_log.csv` — Logs each frame with the following data:
  - ⏱️ **Timestamp**
  - 🎯 **Attention Status** (Attentive / Not Attentive)
  - 📐 **Head Angles**: Pitch, Yaw, Roll
 

**👤 Author**
Vamsi Prasad

