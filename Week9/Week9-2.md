# YOLOv5 소개

## 1. YOLO란 무엇인가요?
YOLO는 **"You Only Look Once"**의 약자로, 이미지를 **그리드 시스템(Grid System)** 으로 나누어 객체를 감지하는 알고리즘입니다.  
그리드의 각 셀은 네트워크 내부에서 물체의 중심을 기준으로 해당 물체를 감지합니다.  
YOLO는 **속도와 정확성**으로 인해 가장 유명한 객체 감지(Object Detection) 알고리즘 중 하나입니다.

---

## 2. YOLOv5란?
YOLOv5는 **Ultralytics**에서 오픈소스로 공개한 YOLO의 버전으로, **PyTorch 기반**으로 완벽히 구현되었습니다.  
YOLOv4에서 다양한 트릭과 실험적 비교가 도입되었지만, YOLOv5는 **더 강력하고, 실시간성이 뛰어나며, 정확한 객체 감지 기술**을 제공합니다.

---

### ✅ 특징

1. **더 빠른 추론 속도**

<img width="1860" height="932" alt="image" src="https://github.com/user-attachments/assets/ee146974-a6cd-4f99-abde-4baaab5423c1" />

   - 공식 수치에 따르면 YOLOv5의 최대 추론 시간은 **이미지당 0.007초**, 즉 **초당 140 FPS**입니다.
   - CPU 환경에서도 이미지 1장당 추론 시간은 **7ms**로, 이는 **140 FPS**를 의미합니다.  
     → 인간의 눈이 요구하는 **20 FPS**보다 훨씬 빠른 속도  
   - 같은 조건에서 YOLOv4는 최대 **50 FPS**까지만 도달 가능  
   - GPU 환경에서는 FPS가 **최대 400**까지 상승

2. **더 작은 모델 크기**

<img width="1053" height="300" alt="image" src="https://github.com/user-attachments/assets/4cc5dc62-7ae1-475e-bf2b-f0eef54bdd51" />

   - YOLOv5의 가중치 파일 크기는 **YOLOv4의 1/9** 수준으로 훨씬 가볍습니다.  
   - 메모리 사용 효율이 뛰어나 다양한 임베디드 환경에 적합합니다.

3. **짧은 학습 시간**
   - 단일 **V100 GPU**에서 COCO 2017 데이터셋 학습 시 YOLOv5는 YOLOv4보다 **짧은 학습 시간**을 보여줍니다.

| YOLOv5 s | YOLOv5 m | YOLOv5 l | YOLOv5 x |
|----------|----------|----------|----------|
|    2     |    4     |    6     |    8     |

---
# Jetson-nano USB-CAM 실습
---

## 1. 개요

본 강의에서는 **NVIDIA Jetson Orin Nano** 환경에서 **YOLOv5 모델**을
활용하여  
USB 카메라 입력을 실시간으로 처리하고, 객체 탐지를 수행하는 방법을
학습합니다.

------------------------------------------------------------------------

## 2. 환경 준비

### 2.1 Jetson Orin Nano 기본 환경 설정

- JetPack (CUDA, cuDNN, TensorRT 포함) 설치 완료
- Python 3.8 이상 권장
- USB 카메라 연결 및 확인

```bash
# 카메라 장치 확인
ls /dev/video*
```

### 2.2 가상환경 생성 및 활성화

패키지 충돌을 막기 위해 Python 가상환경을 사용합니다.

```bash
# venv 패키지가 없다면 설치
sudo apt-get update
sudo apt-get install -y python3.8-venv

# YOLOv5 디렉토리에서 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# pip 최신화
python -m pip install --upgrade pip setuptools wheel
```

### 2.3 필수 라이브러리 설치

YOLOv5 실행을 위해 필요한 Python 라이브러리를 설치합니다.  
(Jetson 환경에서는 최소 의존성 설치 권장)

```bash
# 핵심 패키지
python -m pip install numpy==1.24.4
python -m pip install pillow==10.4.0
python -m pip install PyYAML==6.0.1
python -m pip install requests==2.32.4
python -m pip install tqdm==4.67.1
python -m pip install thop==0.1.1.post2209072238

# 시각화용 패키지
python -m pip install "pandas<2.1" seaborn matplotlib==3.7.5

# OpenCV: GUI로 창을 띄우려면 opencv-python, 저장만 할 경우 headless 버전
python -m pip install opencv-python==4.7.0.72
# python -m pip install opencv-python-headless==4.7.0.72
```

PyTorch/torchvision은 JetPack 버전에 맞는 NVIDIA wheel을 권장하지만,  
테스트용으로는 아래를 사용합니다.

```bash
python -m pip install torch==2.4.1 torchvision==0.19.1
```

------------------------------------------------------------------------

## 3. YOLOv5 설치

### 3.1 GitHub에서 YOLOv5 클론

권장: 안정적인 v6.2 버전 사용

```bash
git clone -b v6.2 https://github.com/ultralytics/yolov5.git
cd yolov5
```

### 3.2 설치 확인

```bash
# 샘플 이미지 탐지
python detect.py --source data/images/bus.jpg --weights yolov5s.pt --conf 0.5
```

------------------------------------------------------------------------

## 4. USB 카메라 연결 및 테스트

### 4.1 OpenCV로 카메라 테스트

별도의 테스트 파일을 작성하여 카메라 입력을 확인합니다.

```bash
nano test_cam.py
```

아래 코드를 입력 후 저장:

```python
import cv2

cap = cv2.VideoCapture(0)  # 0번 카메라 장치

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("USB Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

실행:

```bash
python test_cam.py
```

------------------------------------------------------------------------

## 5. YOLOv5로 실시간 객체 인식

### 5.1 기본 실행

```bash
python detect.py --weights yolov5s.pt --source 0 --conf 0.5
```

- `--weights`: 사용할 모델 가중치 (`yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`, `yolov5x.pt`)
- `--source 0`: 0번 카메라 장치 (USB 카메라)
- `--conf 0.5`: confidence threshold 설정

### 5.2 옵션 설명

- `--img 640` : 입력 이미지 크기 지정
- `--view-img` : 탐지 결과 창 띄우기
- `--save-txt` : 결과를 `.txt`로 저장
- `--save-conf` : confidence score 저장

------------------------------------------------------------------------

## 6. X11 포워딩 및 GUI 실행

Jetson Orin Nano는 보통 모니터 없이 SSH로 접속하므로, 원격에서 GUI 창을 띄우려면 X11 포워딩이 필요합니다.

### 6.1 서버(Jetson) 설정

```bash
sudo apt-get update
sudo apt-get install -y xauth x11-apps
sudo nano /etc/ssh/sshd_config
```

다음 항목 확인 및 수정:

```
X11Forwarding yes
X11UseLocalhost yes
```

SSH 재시작:

```bash
sudo systemctl restart ssh
```

### 6.2 클라이언트(Windows) 설정

- VcXsrv 또는 Xming 설치
- `Xlaunch` 실행 → `Multiple windows`, `Start no client`, `Disable access control` 선택 후 실행

### 6.3 접속 및 테스트

```powershell
ssh -Y ymkoo@<jetson-ip>
```

Jetson에서:

```bash
echo $DISPLAY   # 값이 localhost:10.0 형태면 OK
xclock          # 시계창이 Windows에 뜨면 정상
```

이후 OpenCV 또는 YOLOv5 실행 시 `--view-img` 옵션으로 GUI 창 확인 가능.

------------------------------------------------------------------------

## 7. Xlunch 설치 (옵션)

Jetson에 경량 런처(Xlunch)를 설치하려면 소스 빌드가 필요합니다.

```bash
sudo apt-get install -y git build-essential libx11-dev libxpm-dev libimlib2-dev
git clone https://github.com/Tomas-M/xlunch.git
cd xlunch
make
sudo make install
```

앱 구성 파일(`apps.csv`) 작성 후 실행:

```bash
nano apps.csv
```

예시:

```csv
Terminal,x-terminal-emulator,utilities-terminal
Browser,firefox,web-browser
```

실행:

```bash
xlunch --input apps.csv
```

※ GUI 세션(X11 또는 모니터 환경)이 있어야 실행됩니다.

------------------------------------------------------------------------

## 8. Jetson Orin Nano 최적화

### 8.1 TensorRT 활용

Jetson 환경에서는 TensorRT를 통해 YOLOv5를 최적화할 수 있습니다.

```bash
# TensorRT 변환 예시 (onnx → trt)
python export.py --weights yolov5s.pt --include onnx engine
```

### 8.2 실행 속도 비교

- PyTorch: 초당 약 20~30 FPS
- TensorRT 최적화: 초당 40~60 FPS (환경에 따라 다름)

------------------------------------------------------------------------

## 요약

1. Python 가상환경 생성 및 활성화  
2. YOLOv5 필수 라이브러리 설치  
3. YOLOv5 설치 및 샘플 이미지 테스트  
4. USB 카메라 입력 확인  
5. YOLOv5로 실시간 객체 탐지 실행  
6. X11 포워딩으로 원격 GUI 실행  
7. Xlunch 설치 및 실행 (옵션)  
8. TensorRT 최적화로 FPS 향상

