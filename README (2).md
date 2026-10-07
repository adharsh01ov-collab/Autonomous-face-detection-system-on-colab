# 🧑‍💻 Autonomous Face Recognition & Storage System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FaceNet-EE4C2C.svg)](https://pytorch.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00.svg)](https://colab.research.google.com/)
[![Database](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An **autonomous face detection, recognition and enrollment system** built for **Google Colab**. It detects faces with **MTCNN**, generates 512-dimensional embeddings with **FaceNet (InceptionResnetV1 pretrained on VGGFace2)**, and automatically enrolls unknown faces as `Person_001`, `Person_002`, and so on. Every sighting is logged in an **SQLite** database along with a saved face crop.

> 📄 The full source code is in [`code.md`](code.md).

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [How It Works](#️-how-it-works)
4. [Tech Stack](#-tech-stack)
5. [Project Structure](#-project-structure)
6. [Requirements](#-requirements)
7. [Installation & Setup](#-installation--setup)
8. [Usage](#-usage)
9. [Configuration Parameters](#️-configuration-parameters)
10. [Function Reference](#-function-reference)
11. [Database Schema](#️-database-schema)
12. [Output Screenshots](#️-output-screenshots)
13. [Sample Outputs](#-sample-outputs)
14. [Managing the Database](#️-managing-the-database)
15. [Persisting Data](#-persisting-data)
16. [Running Locally](#-running-locally)
17. [Troubleshooting](#-troubleshooting)
18. [FAQ](#-faq)
19. [Limitations](#️-limitations)
20. [Privacy & Ethics](#-privacy--ethics)
21. [Future Improvements](#-future-improvements)
22. [Contributing](#-contributing)
23. [Author](#-author)
24. [License](#-license)
25. [Acknowledgements](#-acknowledgements)

---

## 🔎 Overview

Most face recognition demos require you to manually register people first. This project is **autonomous**: it starts with an empty database, and whenever it sees a face it does not recognize, it enrolls that person automatically. When the same person appears again, even in a different image, video or webcam frame, the system recognizes them and logs a new sighting.

**Typical use cases**
- Attendance and visitor logging (with consent)
- Smart-camera and IoT experiments
- Learning how face embeddings and similarity matching work
- Prototyping a face gallery without manual labeling

---

## ✨ Features

- 🔍 **Face detection** with MTCNN and a confidence filter
- 🧠 **Face embeddings** using FaceNet (InceptionResnetV1, VGGFace2 weights)
- 🤖 **Autonomous enrollment**: unknown faces are auto-registered as `Person_XXX`
- ✅ **Recognition** using cosine similarity against a stored gallery
- 🗂️ **Persistent storage**: SQLite database plus saved face crops
- 📝 **Sighting log**: every detection is timestamped with source and similarity score
- 🖼️ **Multiple input types**: uploaded images, video files, or live browser webcam
- 🔁 **Adaptive gallery**: stores up to N embeddings per person to capture pose and lighting variety
- 👥 **Multi-face support**: handles several faces in one frame
- 🛠️ **Database tools**: list, rename, delete people, view crops and logs
- ⚡ **GPU acceleration** when available, with automatic CPU fallback

---

## ⚙️ How It Works

```
Input (image / video frame / webcam frame)
        │
        ▼
MTCNN face detection ──► discard detections below MIN_DET_CONF
        │
        ▼
Aligned 160x160 face tensors
        │
        ▼
InceptionResnetV1 ──► 512-D embedding ──► L2 normalization
        │
        ▼
Cosine similarity against every stored embedding (gallery)
        │
        ├── best match ≥ SIM_THRESHOLD ──► KNOWN person
        │        ├── log sighting + save crop
        │        └── add embedding if different enough (up to MAX_EMB_PER_PERSON)
        │
        └── best match < SIM_THRESHOLD ──► NEW person
                 ├── create Person_XXX
                 ├── store first embedding
                 └── log sighting + save crop
```

**Step-by-step**

1. **Detection**: MTCNN finds face bounding boxes and confidence scores. Faces below `MIN_DET_CONF` are ignored.
2. **Alignment & cropping**: each face is extracted as an aligned 160×160 tensor.
3. **Embedding**: InceptionResnetV1 converts each face into a 512-D vector, which is L2-normalized so a dot product equals cosine similarity.
4. **Matching**: the embedding is compared with every embedding in the gallery. The highest similarity decides the identity.
5. **Decision**: if similarity ≥ `SIM_THRESHOLD` the person is known, otherwise a new person is enrolled.
6. **Gallery growth**: for known people, a new embedding is saved only if the face looks different enough (`< EXTRA_EMB_BELOW`), improving robustness to pose and lighting.
7. **Logging**: each detection saves a crop image and a row in the `sightings` table.

---

## 🧰 Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3 |
| Platform | Google Colab (GPU / CPU) |
| Face detection | MTCNN (`facenet-pytorch`) |
| Face embeddings | InceptionResnetV1 (VGGFace2) |
| Deep learning | PyTorch |
| Image / video processing | OpenCV, Pillow, NumPy |
| Database | SQLite3 |
| Data display | Pandas |
| Webcam capture | Browser `getUserMedia` via Colab JavaScript bridge |

---

## 📁 Project Structure

```
face-recognition-system/
│
├── README.md                      # Project documentation (this file)
├── code.md                        # Complete source code, cell by cell
├── face_recognition_colab.ipynb   # Colab notebook (optional)
├── requirements.txt               # Dependencies (optional)
├── LICENSE                        # MIT license
│
├── face_system/                   # Created automatically at runtime
│   ├── faces.db                   # SQLite database
│   └── crops/                     # Saved face crops: <person_id>_<timestamp>.jpg
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

## 📋 Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.9 or higher |
| Runtime | Google Colab (GPU recommended) |
| Browser | Chrome / Edge / Firefox with camera permission (for webcam mode) |
| Libraries | `facenet-pytorch`, `torch`, `torchvision`, `opencv-python`, `pillow`, `numpy`, `pandas` |

Suggested `requirements.txt` for local use:

```
facenet-pytorch
torch
torchvision
opencv-python
pillow
numpy
pandas
```

---

## 🚀 Installation & Setup

### Option 1: Google Colab (recommended)

1. Open [Google Colab](https://colab.research.google.com/) and create a new notebook.
2. *(Optional)* Enable GPU: **Runtime → Change runtime type → T4 GPU**.
3. Copy each block from [`code.md`](code.md) into its own cell.
4. Run the cells from top to bottom.

The install cell uses `--no-deps` so Colab's preinstalled `torch` and `numpy` are not broken:

```python
!pip install -q facenet-pytorch --no-deps
```

Expected output after the setup cell:

```
Device: cuda | DB: /content/face_system/faces.db
```

### Option 2: Clone from GitHub

```bash
git clone https://github.com/<your-username>/face-recognition-system.git
cd face-recognition-system
```

Then upload the notebook to Colab or follow the [Running Locally](#-running-locally) section.

---

## 📖 Usage

### 1️⃣ Recognize uploaded images

```python
run_on_uploads()
```

Select one or more photos. For each image the system prints the result and shows the annotated picture:

- 🟧 **Orange box**: new person enrolled
- 🟩 **Green box**: known person recognized

### 2️⃣ Process a video file

```python
run_on_video()                                          # upload a video
run_on_video('my_video.mp4', every_n=15, show=True)     # use an existing file
```

| Argument | Meaning |
|----------|---------|
| `path` | Video path. If `None`, an upload dialog opens |
| `every_n` | Process every N-th frame (higher = faster, fewer detections) |
| `max_frames` | Stop after this many processed frames |
| `show` | Display annotated frames while processing |

### 3️⃣ Live webcam recognition

```python
run_live(seconds=60, interval=1.0)
```

Allow camera access when the browser prompts. The loop captures a frame every `interval` seconds for `seconds` seconds, recognizes or enrolls faces, and logs sightings.

### 4️⃣ View and manage data

```python
display(list_people())          # all enrolled people
display(sighting_log())         # latest sightings
show_person_crops(1)            # saved crops for Person 1
rename_person(1, 'Adharsh')     # replace Person_001 with a real name
delete_person(2)                # remove a person and all related data
```

---

## 🎛️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `USE_DRIVE` | `False` | `True` mounts Google Drive so data persists across sessions |
| `SIM_THRESHOLD` | `0.65` | Minimum cosine similarity to treat two faces as the same person (typical range 0.55 to 0.75) |
| `MIN_DET_CONF` | `0.95` | Minimum MTCNN confidence to accept a detection |
| `MAX_EMB_PER_PERSON` | `15` | Maximum embeddings stored per person |
| `EXTRA_EMB_BELOW` | `0.85` | A known face adds a new embedding only if its similarity is below this value |

**Tuning guide**

| Symptom | Fix |
|---------|-----|
| Same person gets multiple IDs | Lower `SIM_THRESHOLD` (e.g. 0.60) |
| Different people merged into one ID | Raise `SIM_THRESHOLD` (e.g. 0.70) |
| Blurry or partial faces enrolled | Raise `MIN_DET_CONF` |
| Faces being missed | Lower `MIN_DET_CONF` (e.g. 0.90) |
| Gallery growing too large | Lower `MAX_EMB_PER_PERSON` |

---

## 🧩 Function Reference

| Function | Purpose |
|----------|---------|
| `load_gallery()` | Loads all stored embeddings and their person IDs from the database |
| `person_name(pid)` | Returns the name of a person by ID |
| `new_person()` | Creates a new `Person_XXX` record and returns its ID |
| `add_embedding(pid, emb)` | Stores an embedding for a person |
| `process_frame(img_rgb, source, annotate)` | Core pipeline: detect, embed, match, enroll, log, annotate |
| `run_on_uploads()` | Processes uploaded images |
| `run_on_video(path, every_n, max_frames, show)` | Processes frames from a video |
| `start_camera()` | Starts the browser webcam |
| `grab_frame()` | Captures a single webcam frame as an RGB array |
| `run_live(seconds, interval)` | Autonomous live recognition loop |
| `list_people()` | DataFrame of all people with sighting counts |
| `rename_person(pid, new_name)` | Renames a person |
| `delete_person(pid)` | Deletes a person, embeddings and sightings |
| `sighting_log(limit)` | Latest sightings as a DataFrame |
| `show_person_crops(pid, n)` | Displays the latest saved crops of a person |

---

## 🗄️ Database Schema

Database file: `face_system/faces.db`

### `persons`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK, autoincrement) | Unique person ID |
| `name` | TEXT | Display name, e.g. `Person_001` |
| `created_at` | TEXT | Enrollment timestamp (ISO format) |

### `embeddings`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK, autoincrement) | Row ID |
| `person_id` | INTEGER | Linked person |
| `emb` | BLOB | 512-D float32 embedding (2048 bytes) |

### `sightings`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER (PK, autoincrement) | Row ID |
| `person_id` | INTEGER | Linked person |
| `ts` | TEXT | Sighting timestamp |
| `source` | TEXT | File name or `webcam` |
| `similarity` | REAL | Match score (1.0 for a new enrollment) |
| `crop_path` | TEXT | Path to the saved face crop |

**Relationships**

```
persons (1) ──── (many) embeddings
persons (1) ──── (many) sightings
```

---

## 🖼️ Output Screenshots

### 1. Setup and device info

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2861eb5e-d499-4f0b-9cdf-5411a9806d99" />

*Model loading and database initialization.*

### 2. Image recognition (uploaded photos)


<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/4423d945-62a0-4689-9d05-42551c1b3ec3" />


*Orange box: new person enrolled. Green box: known person recognized.*

### 3. Video processing

https://github.com/user-attachments/assets/c5f03d0e-e548-4067-91a3-71767b992adb


*Faces detected and logged from sampled video frames.*


### 4. People table

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/128f8421-a6d9-4dfb-906a-2f6bf6de8d10" />


*Enrolled people with sighting counts and last-seen time.*


## 🧪 Sample Outputs

Example console outputs (replace with your own results):

**Image recognition**

```
photo1.jpg -> [{'person_id': 1, 'name': 'Person_001', 'status': 'NEW', 'similarity': 1.0}]
photo2.jpg -> [{'person_id': 1, 'name': 'Person_001', 'status': 'known', 'similarity': 0.82}]
group.jpg  -> [{'person_id': 2, 'name': 'Person_002', 'status': 'NEW', 'similarity': 1.0},
               {'person_id': 3, 'name': 'Person_003', 'status': 'NEW', 'similarity': 1.0}]
```

**Video processing**

```
Processed 120 frames from classroom.mp4
```

**Live webcam**

```
['Person_001 (known)']
['Person_001 (known)', 'Person_004 (NEW)']
no face
```

**People table**

| id | name | created_at | sightings | last_seen |
|----|------|------------|-----------|-----------|
| 1 | Person_001 | 2026-10-07T11:40:12 | 14 | 2026-10-07T11:52:30 |
| 2 | Person_002 | 2026-10-07T11:41:03 | 5 | 2026-10-07T11:50:18 |

---

## 🛠️ Managing the Database

```python
list_people()                  # table of all enrolled people
rename_person(1, 'Adharsh')    # give a person a real name
delete_person(3)               # remove a person and all their data
sighting_log(limit=20)         # latest sightings
show_person_crops(1, n=5)      # view saved crops
```

To merge duplicate identities manually, reassign the records with SQL:

```python
db.execute('UPDATE embeddings SET person_id=1 WHERE person_id=5')
db.execute('UPDATE sightings  SET person_id=1 WHERE person_id=5')
db.execute('DELETE FROM persons WHERE id=5')
db.commit()
```

---

## 💾 Persisting Data

By default data is stored in `/content/face_system`, which is **deleted when the Colab session ends**.

**Option A: Google Drive**
Set `USE_DRIVE = True` in the config cell. Drive will be mounted and data saved in `MyDrive/face_system`.

**Option B: Download a backup**

```python
!cd /content && zip -rq face_system.zip face_system
from google.colab import files
files.download('/content/face_system.zip')
```

---

## 💻 Running Locally

The notebook uses Colab-specific modules (`google.colab`). To run on your own machine:

1. Install the requirements:
   ```bash
   pip install facenet-pytorch torch torchvision opencv-python pillow numpy pandas
   ```
2. Remove these imports: `from google.colab import drive, files`, `eval_js`, `cv2_imshow`.
3. Replace `cv2_imshow(img)` with `cv2.imshow('result', img); cv2.waitKey(1)`.
4. Replace the browser webcam code with:
   ```python
   cap = cv2.VideoCapture(0)
   ok, frame = cap.read()
   rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
   out, res = process_frame(rgb, source='webcam')
   ```
5. Replace `files.upload()` with local file paths.
6. Change `BASE` to a local folder such as `./face_system`.

---

## 🩺 Troubleshooting

| Problem | Likely cause | Solution |
|---------|--------------|----------|
| `ModuleNotFoundError: facenet_pytorch` | Install cell not run | Run the install cell and restart the runtime if needed |
| Colab breaks after install | Dependency conflict | Use `--no-deps` as in the install cell |
| Drive `credential propagation` error | Drive mount failed | Keep `USE_DRIVE = False`, or mount Drive manually in a separate cell |
| Webcam does not start | Camera permission denied | Allow camera access in the browser and rerun |
| `no face found` | Small, blurry or dark face | Use a clearer image or lower `MIN_DET_CONF` |
| Same person gets many IDs | Threshold too strict or pose varies | Lower `SIM_THRESHOLD` |
| Two people share one ID | Threshold too loose | Raise `SIM_THRESHOLD` |
| Very slow processing | Running on CPU or too many frames | Enable GPU, increase `every_n`, reduce `max_frames` |
| Data lost after session | Colab storage is temporary | Use Drive or download a backup |

---

## ❓ FAQ

**Does it need internet?**
Only to install packages and download the pretrained weights on first run.

**Is training required?**
No. It uses pretrained models, so no training is needed.

**Can it recognize multiple faces in one image?**
Yes. `MTCNN(keep_all=True)` detects all faces, and each is processed separately.

**Why are new people named `Person_001`?**
The system is autonomous and has no names to begin with. Use `rename_person()` to assign real names.

**How accurate is it?**
FaceNet performs well on frontal, well-lit faces. Accuracy drops with masks, sunglasses, extreme angles and low resolution. Tune `SIM_THRESHOLD` for your data.

---

## ⚠️ Limitations

- Accuracy depends on lighting, pose, resolution and occlusion.
- Auto-enrollment can create **duplicate identities** when the threshold is strict.
- The gallery search is a brute-force comparison, which slows down for very large databases.
- No liveness or anti-spoofing check, so a photo of a face may be accepted.
- Colab storage is temporary unless Drive or a backup is used.
- Not suitable for security-critical or legal identification.

---

## 🔐 Privacy & Ethics

Face data is **biometric and sensitive personal data**.

- Use this system only with the **informed consent** of the people being recorded.
- Follow your local privacy and data protection laws.
- Do not use it for covert surveillance.
- Store and share the database securely, and delete data you no longer need with `delete_person()`.
- This project is intended for **learning and research**.

---

## 🔮 Future Improvements

- [ ] Web dashboard (Streamlit / Flask) for managing identities
- [ ] Duplicate-identity merge tool
- [ ] Liveness / anti-spoofing detection
- [ ] FAISS index for fast search on large galleries
- [ ] CSV export of the sighting log
- [ ] RTSP / IP camera support
- [ ] Face tracking across video frames to reduce duplicate logs
- [ ] Attendance report generation

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 👤 Author

**Adharsh V**
B.E. Electronics & Communication Engineering, Saveetha Engineering College (Anna University)

- GitHub: [@adharsh01ov-collab](https://github.com/adharsh01ov-collab)
- LinkedIn: https://www.linkedin.com/in/adharsh-v-a62a6343b/
- Email: your-email@example.com

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) by Tim Esler
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [MTCNN: Joint Face Detection and Alignment](https://arxiv.org/abs/1604.02878)
- [VGGFace2 Dataset](https://arxiv.org/abs/1710.08092)

---

⭐ If you found this project useful, please give it a star!
