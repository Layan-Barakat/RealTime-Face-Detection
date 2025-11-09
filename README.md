# Real-Time Face Detection

Two minimal webcam apps:
- **Haar** cascade detector (classic + fast)
- **OpenCV DNN (Caffe)** SSD face detector (robust, with FPS overlay)

---

## Structure
```text
.
├─ scripts/
│  ├─ haar_webcam.py
│  └─ caffe_webcam.py
├─ models/
│  ├─ deploy.prototxt
│  └─ res10_300x300_ssd_iter_140000.caffemodel
├─ requirements.txt
└─ README.md
```

> **Note:** The Caffe model files are **not included** in this repo.  
> Download them from OpenCV’s repo and place them in `models/` before running:
> https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector  
> Files needed:
> - `deploy.prototxt`
> - `res10_300x300_ssd_iter_140000.caffemodel`

---

## Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Run
```bash
# Haar cascade
python scripts/haar_webcam.py

# OpenCV DNN (Caffe)
python scripts/caffe_webcam.py
```

Press **Esc** to exit the window.

---

## Author
Layan Barakat — University of Birmingham Dubai
