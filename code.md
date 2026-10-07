# 💻 Source Code: Autonomous Face Recognition & Storage System

Complete code for the Google Colab notebook. Copy each block into its own Colab cell and run them **top to bottom**.

> Tip: Use a GPU runtime for speed (**Runtime → Change runtime type → T4 GPU**). CPU also works.

---

## Cell 1: Install Dependencies

`--no-deps` avoids breaking Colab's preinstalled `torch` / `numpy`.

```python
!pip install -q facenet-pytorch --no-deps
```

---

## Cell 2: Config, Imports & Database

```python
import os, io, base64, sqlite3, time
from datetime import datetime
import numpy as np, cv2, torch
from PIL import Image
from IPython.display import display, Javascript, clear_output
from google.colab import drive, files
from google.colab.output import eval_js
from google.colab.patches import cv2_imshow
from facenet_pytorch import MTCNN, InceptionResnetV1

USE_DRIVE = False         # Set True to mount Google Drive manually (persistent storage)
SIM_THRESHOLD = 0.65      # cosine similarity needed to call two faces the same person (tune 0.55-0.75)
MIN_DET_CONF = 0.95       # ignore weak face detections
MAX_EMB_PER_PERSON = 15   # keep up to N embeddings per person (adds pose/lighting variety)
EXTRA_EMB_BELOW = 0.85    # only store a new embedding if it differs enough from existing ones

if USE_DRIVE:
    drive.mount('/content/drive')
    BASE = '/content/drive/MyDrive/face_system'
else:
    BASE = '/content/face_system'
CROP_DIR = f'{BASE}/crops'
os.makedirs(CROP_DIR, exist_ok=True)
DB_PATH = f'{BASE}/faces.db'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

db = sqlite3.connect(DB_PATH, check_same_thread=False)
db.executescript('''
CREATE TABLE IF NOT EXISTS persons(
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS embeddings(
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, emb BLOB);
CREATE TABLE IF NOT EXISTS sightings(
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, ts TEXT,
    source TEXT, similarity REAL, crop_path TEXT);
''')
db.commit()
print('Device:', device, '| DB:', DB_PATH)
```

---

## Cell 3: Core Logic

```python
def load_gallery():
    rows = db.execute('SELECT person_id, emb FROM embeddings').fetchall()
    if not rows:
        return np.empty(0, dtype=int), np.empty((0, 512), dtype=np.float32)
    ids = np.array([r[0] for r in rows])
    mat = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, mat

def person_name(pid):
    return db.execute('SELECT name FROM persons WHERE id=?', (pid,)).fetchone()[0]

def new_person():
    cur = db.execute('INSERT INTO persons(name, created_at) VALUES(?,?)',
                     ('tmp', datetime.now().isoformat(timespec='seconds')))
    pid = cur.lastrowid
    db.execute('UPDATE persons SET name=? WHERE id=?', (f'Person_{pid:03d}', pid))
    return pid

def add_embedding(pid, emb):
    db.execute('INSERT INTO embeddings(person_id, emb) VALUES(?,?)',
               (pid, emb.astype(np.float32).tobytes()))

def process_frame(img_rgb, source='unknown', annotate=True):
    """img_rgb: numpy RGB array. Returns (annotated BGR image, list of results)."""
    pil = Image.fromarray(img_rgb)
    boxes, probs = mtcnn.detect(pil)
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    results = []
    if boxes is None:
        return out, results

    keep = [i for i, p in enumerate(probs) if p is not None and p >= MIN_DET_CONF]
    if not keep:
        return out, results
    boxes = boxes[keep]
    faces = mtcnn.extract(pil, boxes, None)           # aligned 160x160 tensors
    with torch.no_grad():
        embs = resnet(faces.to(device)).cpu().numpy()
    embs /= np.linalg.norm(embs, axis=1, keepdims=True)

    ids, gallery = load_gallery()
    for box, emb in zip(boxes, embs):
        x1, y1, x2, y2 = [int(max(0, v)) for v in box]
        best_pid, best_sim = None, 0.0
        if len(ids):
            sims = gallery @ emb
            j = int(np.argmax(sims))
            best_pid, best_sim = int(ids[j]), float(sims[j])

        if best_pid is not None and best_sim >= SIM_THRESHOLD:
            pid, status = best_pid, 'known'
            n = int((ids == pid).sum())
            if best_sim < EXTRA_EMB_BELOW and n < MAX_EMB_PER_PERSON:
                add_embedding(pid, emb)
        else:
            pid, status, best_sim = new_person(), 'NEW', 1.0
            add_embedding(pid, emb)

        name = person_name(pid)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        crop_path = f'{CROP_DIR}/{pid}_{ts}.jpg'
        cv2.imwrite(crop_path, out[y1:y2, x1:x2])
        db.execute('INSERT INTO sightings(person_id, ts, source, similarity, crop_path) VALUES(?,?,?,?,?)',
                   (pid, datetime.now().isoformat(timespec='seconds'), source, best_sim, crop_path))
        db.commit()
        results.append({'person_id': pid, 'name': name, 'status': status, 'similarity': round(best_sim, 3)})

        if annotate:
            color = (0, 200, 0) if status == 'known' else (0, 140, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(out, f'{name} ({best_sim:.2f})', (x1, max(15, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if status == 'NEW':  # refresh gallery so the same person in this frame isn't enrolled twice
            ids, gallery = load_gallery()
    return out, results
```

---

## Cell 4a: Process Uploaded Images

Run the cell, then pick one or more photos.

```python
def run_on_uploads():
    up = files.upload()
    for fname, data in up.items():
        img = np.array(Image.open(io.BytesIO(data)).convert('RGB'))
        out, res = process_frame(img, source=fname)
        print(fname, '->', res if res else 'no face found')
        cv2_imshow(out)

run_on_uploads()
```

---

## Cell 4b: Process a Video File

Upload a video or pass a path. Samples every N-th frame.

```python
def run_on_video(path=None, every_n=15, max_frames=300, show=False):
    if path is None:
        up = files.upload()
        path = list(up.keys())[0]
    cap = cv2.VideoCapture(path)
    i = done = 0
    while cap.isOpened() and done < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if i % every_n == 0:
            out, res = process_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), source=os.path.basename(path))
            done += 1
            if show and res:
                clear_output(wait=True); cv2_imshow(out)
        i += 1
    cap.release()
    print(f'Processed {done} frames from {path}')

# run_on_video()   # uncomment to use
```

---

## Cell 4c: Live Webcam (Browser Camera)

Allow camera access when your browser prompts you.

```python
def start_camera():
    display(Javascript('''
    async function startCam(){
      const video = document.createElement('video');
      video.style.display='block'; video.width=320;
      const stream = await navigator.mediaDevices.getUserMedia({video:true});
      document.body.appendChild(video);
      video.srcObject = stream; await video.play();
      window.__cam = video;
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);
    }
    window.startCam = startCam;
    '''))
    eval_js('startCam()')

def grab_frame():
    data = eval_js('''(function(){
      const v = window.__cam, c = document.createElement('canvas');
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext('2d').drawImage(v, 0, 0);
      return c.toDataURL('image/jpeg', 0.8);})()''')
    arr = np.frombuffer(base64.b64decode(data.split(',')[1]), np.uint8)
    return cv2.cvtColor(cv2.imdecode(arr, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

def run_live(seconds=60, interval=1.0):
    """Autonomous loop: grabs a frame every `interval` s, recognizes/enrolls, logs."""
    start_camera()
    t_end = time.time() + seconds
    while time.time() < t_end:
        out, res = process_frame(grab_frame(), source='webcam')
        clear_output(wait=True)
        cv2_imshow(out)
        print([f"{r['name']} ({r['status']})" for r in res] or 'no face')
        time.sleep(interval)
    print('Live session finished.')

# run_live(seconds=60)   # uncomment to use
```

---

## Cell 5: Manage the Database

```python
import pandas as pd

def list_people():
    return pd.read_sql_query('''
        SELECT p.id, p.name, p.created_at,
               COUNT(s.id) AS sightings, MAX(s.ts) AS last_seen
        FROM persons p LEFT JOIN sightings s ON s.person_id = p.id
        GROUP BY p.id ORDER BY p.id''', db)

def rename_person(pid, new_name):
    db.execute('UPDATE persons SET name=? WHERE id=?', (new_name, pid)); db.commit()

def delete_person(pid):
    for t, col in (('embeddings', 'person_id'), ('sightings', 'person_id'), ('persons', 'id')):
        db.execute(f'DELETE FROM {t} WHERE {col}=?', (pid,))
    db.commit()

def sighting_log(limit=20):
    return pd.read_sql_query(f'''
        SELECT s.ts, p.name, s.source, ROUND(s.similarity,3) AS sim
        FROM sightings s JOIN persons p ON p.id = s.person_id
        ORDER BY s.id DESC LIMIT {limit}''', db)

def show_person_crops(pid, n=5):
    rows = db.execute('SELECT crop_path FROM sightings WHERE person_id=? ORDER BY id DESC LIMIT ?', (pid, n)).fetchall()
    for (p,) in rows:
        img = cv2.imread(p)
        if img is not None:
            cv2_imshow(cv2.resize(img, (160, 160)))

display(list_people())
# rename_person(1, 'Adharsh')
# display(sighting_log())
# show_person_crops(1)
```

---

## Optional: Download Your Data

Since Colab storage is temporary, zip and download the database and crops before closing the session.

```python
!cd /content && zip -rq face_system.zip face_system
from google.colab import files
files.download('/content/face_system.zip')
```
