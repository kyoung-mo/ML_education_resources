# YOLO로 배우는 '컴퓨터 비전'

---

## [1] 강의 개요
본 과정은 실습 및 실무 중심으로 구성되어 있습니다.

### 1.1. YOLO 와 컴퓨터 비전
#### 1.1.1. 컴퓨터 비전
- 컴퓨터 비전은 컴퓨터가 인간의 시각적 인식을 모방하여 이미지나 비디오에서 의미 있는 정보를 추출하고 해석하는 기술 분야.
- 최근 딥러닝/머신러닝을 활용하여 객체 인식, 분류, 추적 등을 수행.
- 자율주행, 얼굴 인식, 의료 영상 분석 등 다양한 분야에서 핵심 역할.

#### 1.1.2. YOLO
- YOLO(You Only Look Once)는 실시간 객체 탐지를 위한 딥러닝 알고리즘.
- 이미지 전체를 단일 패스로 분석하여 객체의 위치와 종류를 동시에 예측.
- R-CNN 계열처럼 여러 번 분석하지 않고 원본 이미지를 그대로 처리.

#### 1.1.3. YOLO의 장점
- **빠른 속도**: 단일 패스(single pass)로 처리 → 실시간 애플리케이션에 적합.  
- **높은 정확도**: 빠른 속도에도 다양한 객체를 높은 정확도로 탐지.  
- **다양한 작업**: 탐지뿐 아니라 분류, 분할, 자세 추정 등으로 확장 가능.  

#### 1.1.4. Ultralytics
- YOLO v1~v4: Darknet 프로젝트 주도.  
- YOLO v5~v8(현재): Ultralytics 주도.  
- 사용자 친화적 인터페이스 및 기능 제공 → 모델 훈련/배포 용이.  

#### 1.1.5. Tasks
YOLO는 다양한 컴퓨터 비전 작업을 지원:
- **Object Detection**: 바운딩 박스로 객체 탐지.  
- **Instance Segmentation**: 객체 윤곽선 파악, 픽셀 단위 영역 분할.  
- **Classification**: 이미지 전체 분류(예: 고양이, 개, 자동차).  
- **Pose Estimation / Keypoints Detection**: 관절/포인트 추적 → 자세 분석.  
- **Oriented Bounding Boxes (OBB)**: 회전된 객체 탐지 (차량, 책, 가방 등).  

---

### 1.2. 인공지능 기초
#### 1.2.1. 머신러닝과 딥러닝의 작동원리 기초
- 머신러닝 개요, 기계 학습 방법, 하이퍼파라미터  
- 딥러닝과 DNN, CNN, DeepRacer 모델, 딥러닝 이슈  

#### 1.2.2. YOLO의 작동원리 기초
- **그리드 분할**: S × S 그리드 셀로 이미지 나눔.  
- **빠른 이유**: 1-stage 구조 → RoI 추출 단계 없음.  
- **작은 객체 약점**: 그리드 셀 대비 작은 물체는 검출 어려움.  
- **출력 구조**:  
  - Localization: 5 × B (x, y, w, h, confidence)  
  - Classification: C (클래스 개수)  
- **NMS(Non-Maximum Suppression)**: 중복 박스 제거, 신뢰도 높은 상자 유지.  

---

### 1.3. Colab (모델 훈련)
#### 1.3.1. 모델 훈련 및 추론
- **실행 환경**: Colab (Python3, Jupyter Notebook), 또는 Google Drive → Colab Notebook 생성.  
- **런타임 선택**: [런타임 > 런타임 유형 변경]  
- **컴퓨팅 자원**:  
  - CPU: ~0.1 / 시간  
  - T4 GPU: ~1.6 / 시간 (RAM 13GB, GPU RAM 15GB)  
  - L4 GPU (추천): ~3 / 시간 (RAM 53GB, GPU RAM 23GB)  
  - A100 GPU: ~11 / 시간 (RAM 83GB, GPU RAM 40GB)  
  - TPU v2-8: ~1.8 / 시간 (YOLO 호환성 문제)  
- **최대 런타임**: 12시간 (인터렉션 필요).  

---

### 1.4. Roboflow (라벨링)
#### 1.4.1. Roboflow
- 컴퓨터 비전 프로젝트용 데이터셋 준비, 라벨링, 전처리, 증강 등을 지원하는 플랫폼.  

#### 1.4.2. 주요 기능
- AI 보조 라벨링  
- 데이터셋 관리 및 버전 관리  
- 커스텀 증강(Augmentation)  
- 협업 기능  

#### 1.4.3. 가격 정책
- 일부 유료  
- Starter Plan 2주 무료 제공  

#### 1.4.4. Roboflow Universe
- 다양한 데이터셋 및 사전학습 모델 공유 오픈소스 플랫폼.  
- 데이터 증강 및 커뮤니티 기여 가능.  

#### 1.4.5. 시작하기
1. 사이트 접속: [roboflow.com](https://roboflow.com/)  
2. 가입: 구글, Github, 이메일 계정 가능  
3. 워크스페이스 생성, 멤버 초대 (무료 플랜은 admin 역할만 초대 가능).  

---

### 1.5. 실습 데이터
#### 1.5.1. 데이터 소개
- 목표: DeepRacer Vision Timer와 유사한 모델 제작.  
- 데이터: AWS Deepracer 2023 Sydney 대회 영상 편집본.  

#### 1.5.2. 데이터 구조
- **data 폴더**: 협업을 위해 data_1.mp4, data_2.mp4, data_3.mp4  
- **test.mp4**: 모델 훈련 후 성능 확인용.  

---

### 1.6. 기타 학습 링크
- [IoU와 mAP](https://lynnshin.tistory.com/48)  
- [오토라벨링과 액티브 러닝](https://ahha.ai/2023/12/19/autolabeling/)  
- [YOLO의 원리](https://www.youtube.com/watch?v=L0tzmv--CGY)  

