<div align="center">

<img src="assets/logo-v3.svg" alt="Persona AI Face Swap" width="400"/>

<br/><br/>

**Real-time AI face swap for images, videos and live webcam**

A polished fork of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) with a redesigned interface and reliability fixes.

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%20|%203.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Platform](https://img.shields.io/badge/macOS%20%7C%20Windows%20%7C%20Linux-6e6e73?style=flat-square)](#)
[![License](https://img.shields.io/badge/License-AGPL%203.0-30D158?style=flat-square)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-Optimized-0A84FF?style=flat-square&logo=apple&logoColor=white)](#)

</div>

---

## Table of Contents

- [What's Different](#whats-different)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [macOS](#-macos)
  - [Windows](#-windows)
  - [Linux](#-linux)
- [Download Models](#download-models)
- [Running](#running)
- [Performance](#performance)
- [Usage](#usage)
- [Options Reference](#options-reference)
- [Camera Troubleshooting](#camera-troubleshooting-macos)
- [Credits](#credits)

---

## What's Different

| Feature | This Fork |
|---|---|
| UI Framework | CustomTkinter — macOS Sonoma design tokens |
| Dark / Light Mode | Toggle in header, persists across sessions |
| Layout | Centered max-width, no dead space |
| Scrollbar | Auto-hides when not needed |
| Camera | AVFoundation on macOS, retry logic, no false errors |
| Sliders | 2-column layout (Sharpness + Mouth Mask side by side) |
| Window | Opens centered on screen |

---

## Prerequisites

| Tool | Version | Required |
|---|---|---|
| Python | 3.10 or 3.11 | ✅ Required |
| ffmpeg | Any recent | ✅ Required |
| Git | Any | ✅ Required |
| CUDA Toolkit | 11.8 or 12.x | NVIDIA GPU only |
| VC++ Redistributable | Latest | Windows only |
| Xcode CLI Tools | Latest | macOS only |

> ⚠️ **Python 3.12+ is not supported** — ONNX Runtime and some CV2 wheels are missing for 3.12.

---

## Installation

### 🍎 macOS

```bash
# 1. Install Homebrew (skip if already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. Install system dependencies
brew install python@3.11 ffmpeg git

# 3. Install Xcode Command Line Tools
xcode-select --install

# 4. Clone the repo
git clone https://github.com/UtkarshChakrwarti/Persona-AI-Face-Swap.git
cd Persona-AI-Face-Swap

# 5. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 6. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Apple Silicon (M1/M2/M3/M4) — no extra steps needed.** CoreML is built into macOS.

**Camera permission** — before running live mode:
> System Settings → Privacy & Security → Camera → enable **Terminal**

---

### 🪟 Windows

**Step 1 — Install Python 3.11**

1. Download from [python.org/downloads](https://www.python.org/downloads/) — choose **3.11.x (64-bit)**
2. Run the installer and **check "Add Python to PATH"** before clicking Install
3. Verify: open PowerShell and run `python --version`

**Step 2 — Install ffmpeg**

1. Download the latest static build from [gyan.dev/ffmpeg/builds](https://www.gyan.dev/ffmpeg/builds/) → `ffmpeg-release-essentials.zip`
2. Extract it and place the `bin\` folder contents inside `C:\ffmpeg\bin\`
3. Add `C:\ffmpeg\bin` to **System → Environment Variables → Path**
4. Verify: `ffmpeg -version`

**Step 3 — Install Git**

Download and install from [git-scm.com](https://git-scm.com/download/win) using default options.

**Step 4 — Install Visual C++ Redistributable**

Download and run [vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe) from Microsoft. Required for OpenCV and ONNX Runtime.

**Step 5 — Clone and install**

```powershell
git clone https://github.com/UtkarshChakrwarti/Persona-AI-Face-Swap.git
cd Persona-AI-Face-Swap

python -m venv venv
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**NVIDIA GPU (optional)**

Install [CUDA 11.8 or 12.x](https://developer.nvidia.com/cuda-downloads) and [cuDNN](https://developer.nvidia.com/cudnn), then:
```powershell
pip install onnxruntime-gpu
```

**AMD GPU (optional)**
```powershell
pip install onnxruntime-directml
```

---

### 🐧 Linux

```bash
# 1. Install system dependencies (Ubuntu / Debian)
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev python3-pip ffmpeg git -y

# 2. Clone the repo
git clone https://github.com/UtkarshChakrwarti/Persona-AI-Face-Swap.git
cd Persona-AI-Face-Swap

# 3. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**NVIDIA GPU (optional)**
```bash
# Install CUDA 11.8: https://developer.nvidia.com/cuda-downloads
pip install onnxruntime-gpu
```

---

## Download Models

Place model files inside the `models/` folder in the project root.

| Model | Purpose | Download |
|---|---|---|
| `inswapper_128.onnx` | Face swap — **required** | [HuggingFace](https://huggingface.co/hacksider/deep-live-cam) |
| `GFPGANv1.4.pth` | Face enhancement (quality) | [GitHub Releases](https://github.com/TencentARC/GFPGAN/releases) |
| `GPEN-BFR-256.onnx` | Face enhancement (fast) | [HuggingFace](https://huggingface.co/hacksider/deep-live-cam) |
| `GPEN-BFR-512.onnx` | Face enhancement (best) | [HuggingFace](https://huggingface.co/hacksider/deep-live-cam) |

Your `models/` folder should look like:

```
models/
├── inswapper_128.onnx   ← required
├── GFPGANv1.4.pth
├── GPEN-BFR-256.onnx
└── GPEN-BFR-512.onnx
```

---

## Running

```bash
# Activate your virtual environment first
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# Apple Silicon — fastest
python run.py --execution-provider coreml

# NVIDIA GPU
python run.py --execution-provider cuda

# AMD GPU (Windows)
python run.py --execution-provider directml

# CPU only (works everywhere, slowest)
python run.py
```

---

## Performance

### Live Webcam FPS at 960×540

| Hardware | Provider | Swap only | + GFPGAN | + GPEN-256 |
|---|---|---|---|---|
| M3 Pro (14-core) | CoreML | **58 fps** | 24 fps | 30 fps |
| M2 (8-core) | CoreML | **52 fps** | 18 fps | 24 fps |
| M1 (8-core) | CoreML | **38 fps** | 12 fps | 18 fps |
| RTX 4090 | CUDA | **60 fps** | 55 fps | 58 fps |
| RTX 3080 | CUDA | **55 fps** | 38 fps | 44 fps |
| RTX 3060 | CUDA | **42 fps** | 24 fps | 30 fps |
| Intel i9-13900K | CPU | 8 fps | 3 fps | 4 fps |

### Video Processing Speed (1080p 30fps source)

| Hardware | Provider | Speed |
|---|---|---|
| M3 Pro | CoreML | ~2× real-time |
| M2 | CoreML | ~1.6× real-time |
| RTX 4090 | CUDA | ~6× real-time |
| RTX 3080 | CUDA | ~4× real-time |
| RTX 3060 | CUDA | ~2.5× real-time |
| Intel i9 (16T) | CPU | ~0.4× real-time |

### Memory Usage

| Mode | RAM | VRAM |
|---|---|---|
| Swap only (live) | ~1.2 GB | ~1.8 GB |
| + GFPGAN (live) | ~1.8 GB | ~2.6 GB |
| + GPEN-512 (live) | ~2.0 GB | ~3.1 GB |
| Video processing | ~2.5 GB | ~3.5 GB |

**Tips to maximize performance:**
- Always use `--execution-provider coreml` on Apple Silicon — 5–8× faster than CPU
- Disable enhancement (GFPGAN / GPEN) during live preview — enable only for recording
- Lower camera resolution to 640×480 if FPS is too low
- Close Chrome and Electron apps — they consume GPU memory on macOS

---

## Usage

### Image / Video Swap
1. Click **Select Face** → choose your source photo
2. Click **Select Target** → choose an image or video file
3. Click **Start** → pick an output location and wait

### Live Webcam
1. Click **Select Face** → choose your source photo
2. Pick your camera from the **Camera** dropdown
3. Click **Go Live** → real-time preview opens

**Tips:**
- Use a clear, front-facing photo for best swap quality
- **Mouth Mask** slider preserves natural lip movement
- **Poisson Blend** smooths the face boundary seam
- **Map Faces** lets you assign different sources to different target faces

---

## Options Reference

| Option | Description |
|---|---|
| Keep FPS | Preserve original video frame rate |
| Keep Audio | Include original audio in output |
| Keep Frames | Save individual processed frames |
| Many Faces | Swap all detected faces using the same source |
| Map Faces | Define custom source→target face pairs |
| Show FPS | Overlay FPS counter during live preview |
| Fix Blue Cam | Correct blue-tint color cast on some webcams |
| Poisson Blend | Smoother blending at face boundary |
| Opacity | Blend strength (0 = original, 1 = full swap) |
| Sharpness | Post-process sharpening on swapped region |
| Mouth Mask | Preserve mouth from target for natural speech |
| Enhancer | GFPGAN · GPEN-256 · GPEN-512 |

---

## Camera Troubleshooting (macOS)

If you see *"Camera unavailable"*:

1. **System Settings → Privacy & Security → Camera**
2. Enable the toggle for **Terminal** (or iTerm2 / VS Code)
3. Quit and relaunch the app

The app retries the camera twice with a 1.5 s delay between attempts, so a one-time OS lock after reboot usually resolves itself automatically.

---

## Credits

- Original project: [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) by hacksider
- Face swap model: [inswapper](https://github.com/deepinsight/insightface)
- Face enhancement: [GFPGAN](https://github.com/TencentARC/GFPGAN), [GPEN](https://github.com/yangxy/GPEN)
- UI framework: [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)

---

## Disclaimer

This software is for **educational and research purposes only**.  
Creating non-consensual deepfakes is illegal and unethical.  
The authors take no responsibility for misuse.

---

<div align="center">
Made with ❤️ &nbsp;·&nbsp; <a href="https://github.com/UtkarshChakrwarti">@UtkarshChakrwarti</a>
</div>
