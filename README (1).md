# 🧑‍💻 Autonomous Face Recognition & Storage System

An autonomous face detection, recognition and enrollment system built for **Google Colab**. It detects faces with **MTCNN**, generates 512-D embeddings with **FaceNet (InceptionResnetV1, VGGFace2)**, and automatically enrolls unknown faces as `Person_001`, `Person_002`, … while logging every sighting to an **SQLite** database.

---

## 📌 Table of Contents

- [Features](#-features)
- [How It Works](#-how-it-works)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#️-configuration)
- [Database Schema](#-database-schema)
- [Output Screenshots](#-output-screenshots)
- [Managing the Database](#-managing-the-database)
- [Limitations & Privacy](#️-limitations--privacy)
- [Future Improvements](#-future-improvements)
- [Author](#-author)
- [License](#-license)

---

## ✨ Features

- 🔍 **Face detection** using MTCNN with a confidence filter
- 🧠 **Face embeddings** using FaceNet (InceptionResnetV1 pretrained on VGGFace2)
- 🤖 **Autonomous enrollment**: unknown faces are auto-registered as `Person_XXX`
- ✅ **Recognition** of known faces using cosine similarity
- 🗂️ **Persistent storage**: SQLite database + saved face crops
- 📝 **Sighting log**: every detection is timestamped with source and similarity score
- 🖼️ **Multiple inputs**: uploaded images, video files, or live browser webcam
- 🔁 **Adaptive gallery**: stores up to N embeddings per person for pose/lighting variety
- 🛠️ **Database management**: rename, delete, list people and view crops

---

## ⚙️ How It Works

```
Input (image / video / webcam)
        │
        ▼
 MTCNN face detection  ──► filter by confidence (MIN_DET_CONF)
        │
        ▼
 Aligned 160x160 face crops
        │
        ▼
 InceptionResnetV1 ──► 512-D L2-normalized embedding
        │
        ▼
 Cosine similarity vs. gallery
        │
        ├── ≥ SIM_THRESHOLD ──► Known person  (log sighting, maybe add embedding)
        │
        └── < SIM_THRESHOLD ──► New person    (auto-enroll Person_XXX, log sighting)
```

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Platform | Google Colab (GPU/CPU) |
| Face Detection | MTCNN (`facenet-pytorch`) |
| Face Embeddings | InceptionResnetV1 (VGGFace2) |
| Deep Learning | PyTorch |
| Image/Video Processing | OpenCV, Pillow |
| Database | SQLite |
| Data Display | Pandas |

---

## 📁 Project Structure

```
face-recognition-system/
│
├── face_recognition_colab.ipynb   # Main notebook (or .py exported from Colab)
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies (optional)
│
├── face_system/                   # Created at runtime
│   ├── faces.db                   # SQLite database
│   └── crops/                     # Saved face crops (PersonID_timestamp.jpg)
│
└── screenshots/                   # Output images used in this README
    ├── 01_setup.png
    ├── 02_image_recognition.png
    ├── 03_video_processing.png
    ├── 04_live_webcam.png
    ├── 05_people_table.png
    ├── 06_sighting_log.png
    └── 07_face_crops.png
```

---

## 🚀 Getting Started

### Option 1: Run in Google Colab (Recommended)

1. Upload the notebook/script to [Google Colab](https://colab.research.google.com/)
2. *(Optional)* Switch to GPU: **Runtime → Change runtime type → T4 GPU**
3. Run the cells from top to bottom

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/<your-username>/face-recognition-system.git
cd face-recognition-system

# Install dependencies
pip install facenet-pytorch torch torchvision opencv-python pillow numpy pandas
```

> ⚠️ The webcam and upload features use Google Colab APIs (`google.colab`). For local use, replace them with `cv2.VideoCapture(0)` and local file paths.

---

## 📖 Usage

### 1️⃣ Install & Setup
Run the install cell and the config/database cell. Expected output:

```
Device: cuda | DB: /content/face_system/faces.db
```

### 2️⃣ Recognize Uploaded Images

```python
run_on_uploads()
```
Pick one or more photos. Each face is recognized or enrolled automatically.

### 3️⃣ Process a Video File

```python
run_on_video()                                   # upload a video
run_on_video('my_video.mp4', every_n=15, show=True)
```

### 4️⃣ Live Webcam

```python
run_live(seconds=60, interval=1.0)
```
Allow camera access when your browser prompts you.

### 5️⃣ View & Manage People

```python
display(list_people())
display(sighting_log())
show_person_crops(1)
rename_person(1, 'Adharsh')
delete_person(2)
```

---

## 🎛️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_DRIVE` | `False` | Set `True` to mount Google Drive and persist data |
| `SIM_THRESHOLD` | `0.65` | Cosine similarity to match a face (tune 0.55–0.75) |
| `MIN_DET_CONF` | `0.95` | Minimum MTCNN detection confidence |
| `MAX_EMB_PER_PERSON` | `15` | Max embeddings stored per person |
| `EXTRA_EMB_BELOW` | `0.85` | Store a new embedding only if similarity is below this |

**Tuning tips**
- Too many duplicate people → **lower** `SIM_THRESHOLD`
- Different people merged into one → **raise** `SIM_THRESHOLD`

---

## 🗄️ Database Schema

**`persons`**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Unique person ID |
| name | TEXT | e.g. `Person_001` |
| created_at | TEXT | Enrollment time |

**`embeddings`**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Row ID |
| person_id | INTEGER | Linked person |
| emb | BLOB | 512-D float32 embedding |

**`sightings`**
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER (PK) | Row ID |
| person_id | INTEGER | Linked person |
| ts | TEXT | Timestamp |
| source | TEXT | File name / `webcam` |
| similarity | REAL | Match score |
| crop_path | TEXT | Path to saved face crop |

---

## 🖼️ Output Screenshots

> Replace each placeholder below with your own screenshot from Colab.
> Save images in the `screenshots/` folder and keep the file names (or update the paths).

### 1. Setup & Device Info
<!-- Screenshot of: "Device: cuda | DB: ..." -->
![Setup Output](screenshots/01_setup.png)

*Add caption: Model loading and database initialization.*

### 2. Image Recognition (Uploaded Photos)
<!-- Screenshot of annotated image with bounding box + name + similarity -->
![Image Recognition Output](screenshots/02_image_recognition.png)

*Add caption: Orange box = NEW person enrolled, Green box = known person recognized.*

### 3. Video Processing
<!-- Screenshot of video frame processing / "Processed N frames" message -->
![Video Processing Output](screenshots/03_video_processing.png)

*Add caption: Faces detected and logged from sampled video frames.*

### 4. Live Webcam Recognition
<!-- Screenshot of live webcam frame with detections -->
![Live Webcam Output](screenshots/04_live_webcam.png)

*Add caption: Real-time recognition from browser camera.*

### 5. People Table
<!-- Screenshot of display(list_people()) -->
![People Table](screenshots/05_people_table.png)

*Add caption: Enrolled people with sighting counts and last seen time.*

### 6. Sighting Log
<!-- Screenshot of display(sighting_log()) -->
![Sighting Log](screenshots/06_sighting_log.png)

*Add caption: Timestamped log of recent detections.*

### 7. Saved Face Crops
<!-- Screenshot of show_person_crops(1) -->
![Face Crops](screenshots/07_face_crops.png)

*Add caption: Stored face crops for a person.*

---

## 🛠️ Managing the Database

```python
list_people()                  # table of all enrolled people
rename_person(1, 'Adharsh')    # give a person a real name
delete_person(3)               # remove a person and all their data
sighting_log(limit=20)         # latest sightings
show_person_crops(1, n=5)      # view saved crops
```

---

## ⚠️ Limitations & Privacy

- Accuracy depends on lighting, pose, image quality and occlusion (masks, sunglasses).
- Auto-enrollment can create **duplicate identities** if the threshold is too strict.
- Data on Colab is **temporary** unless `USE_DRIVE = True`.
- Face data is **biometric and sensitive**. Use this system **only with the consent** of the people being recorded and comply with local privacy laws. Do not use it for surveillance without permission.

---

## 🔮 Future Improvements

- [ ] Web dashboard (Streamlit / Flask) for managing identities
- [ ] Merge-duplicates tool for identity clean-up
- [ ] Face anti-spoofing / liveness detection
- [ ] Faster vector search (FAISS) for large galleries
- [ ] Export sighting logs to CSV
- [ ] Real-time RTSP / IP camera support

---

## 👤 Author

**Adharsh V**
B.E. Electronics & Communication Engineering, Saveetha Engineering College

- GitHub: [@your-username](https://github.com/your-username)
- LinkedIn: [your-linkedin](https://linkedin.com/in/your-linkedin)
- Email: your-email@example.com

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Tim Esler
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [VGGFace2 Dataset](https://arxiv.org/abs/1710.08092)

⭐ If you found this project useful, consider giving it a star!
