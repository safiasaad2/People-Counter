# 🧍‍♀️ People ROI Entry/Exit Counter

### 🎯 Overview

This project detects and counts people entering and exiting a defined **Region of Interest (ROI)** using **YOLOv8** for detection and **ByteTrack** for tracking.
It’s perfect for analyzing **crowd flow** or **entry/exit counts** in any monitored area — such as **store entrances**, **corridors**, or **general scenes**.

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Libraries](https://img.shields.io/badge/Libraries-Ultralytics%20%7C%20Supervision%20%7C%20ByteTrack%20%7C%20OpenCV%20%7C%20NumPy-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Active-brightgreen)]()


---

## 🎥 Output Preview

<p align="center">
  <img src="demo.gif" alt="People ROI Counter Demo" width="700">
</p>

---

## 🚀 Features

✅ Real-time detection using **YOLOv8**
<br>
✅ Tracking with **ByteTrack** for consistent ID assignment
<br>
✅ Dynamic **IN/OUT** counting based on motion direction
<br>
✅ Supports both **rectangular** and **rotated** ROI regions
<br>
✅ Saves processed **output video** with live statistics overlay

---

## 🧩 Requirements

Install dependencies before running:

```bash
pip install ultralytics supervision opencv-python numpy
```

---

## 🧠 How It Works

1. **YOLOv8** detects people in each video frame.
2. **ByteTrack** assigns consistent IDs to track each person.
3. When a tracked person crosses into or out of the ROI, the system counts them as **IN** or **OUT** based on movement direction.
4. The total number of people inside the ROI is updated continuously.

---

## 📊 Output Metrics

| Metric               | Meaning                                 |
| -------------------- | --------------------------------------- |
| **IN (front→back)**  | Number of people who entered            |
| **OUT (back→front)** | Number of people who exited             |
| **TOTAL (IN–OUT)**   | Current number of people inside the ROI |

---

## ⚙️ Usage

Run the script with your input video and ROI coordinates:

```bash
python people_roi_counter.py --video input.mp4 --roi "0.20,0.65,0.75,0.70"
```

### Optional Arguments

| Argument       | Description                                       | Example                          |
| -------------- | ------------------------------------------------- | -------------------------------- |
| `--video`      | Path to the input video                           | `--video people.mp4`             |
| `--output`     | Output file path                                  | `--output output_people_roi.mp4` |
| `--model`      | YOLOv8 model name or path                         | `--model yolov8m.pt`             |
| `--roi`        | ROI rectangle as percentages (x1,y1,x2,y2)        | `--roi "0.2,0.6,0.7,0.8"`        |
| `--roi_rot`    | Rotated ROI: center (cx,cy), width, height, angle | `--roi_rot "0.5,0.5,0.3,0.1,25"` |
| `--no-display` | Disable on-screen video display                   | `--no-display`                   |

---

## 🧮 Example Command

```bash
python people_roi_counter.py --video people.mp4 --output output_people_roi.mp4 --roi "0.20,0.65,0.75,0.70"
```

This will:

* Detect people in the video
* Track them as they move across the ROI
* Count entries and exits
* Display live stats and save the annotated output

---

## 📊 Output Video

After processing, an annotated video named **`output_people_roi.mp4`** will be saved automatically.
The output includes:

🎯 Bounding boxes for each tracked person
<br>
🟩 ROI region visualization
<br>
📈 Entry/Exit counters displayed on-screen
<br>
🧭 Movement trails showing direction

---

## 📁 Project Structure

```
People-Counter/
│
├── people_roi_counter.py      # 🎯 Main detection & counting script
├── requirements.txt           # 📦 Required Python dependencies
├── README.md                  # 📘 Project documentation and visuals
├── LICENSE                    # ⚖️ License file (MIT)
├── .gitignore                 # 🚫 Files and folders to ignore in Git
├── input.mp4                  # 🎥 Input video (optional)
└── output_people_roi.mp4      # 💾 Processed output video

```

---

## 🧠 Tips

💡 Adjust the ROI to fit your camera’s angle or gate position.
<br>
💡 Use static camera videos for best results.
<br>
💡 Try different YOLOv8 models (`yolov8n.pt`, `yolov8s.pt`, `yolov8m.pt`) to balance speed vs. accuracy.
<br>
💡 Works with various scenarios such as **store entrances**, **mall doors**, or **building entries**.

---

## 🧰 Notes

* Works best with **static cameras** (no movement).
* The ROI can be customized per scene for accuracy.
* You can freely switch YOLOv8 model sizes depending on hardware capability.

---

## 💬 Credits

Developed using:

* [**YOLOv8** (Ultralytics)](https://github.com/ultralytics/ultralytics)
* [**ByteTrack** integration from Supervision](https://github.com/roboflow/supervision)
* [**OpenCV**](https://opencv.org/) for visualization and video processing

---

## 👩‍💻 Author

**Safia Saad**
💼 AI Engineer | Computer Vision & Deep Learning Enthusiast
📧 [safiakotb123@gmail.com](mailto:safiakotb123@gmail.com)
🌐 [LinkedIn](https://www.linkedin.com/in/safia-saad/)

---

## 🏷️ License

This project is licensed under the **MIT License** — you are free to use, modify, and distribute it.

---
