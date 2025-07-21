---
# 저번 주차 과제
---
# Optimizer란?

머신러닝에서 **Optimizer(최적화 알고리즘)** 는 모델의 **손실 함수(loss function)** 를 최소화하는 방향으로 **가중치(parameter)를 업데이트**해주는 역할을 한다.  
이는 보통 **경사하강법(Gradient Descent)** 알고리즘을 기반으로 하며, 모델이 더 나은 예측을 하도록 학습을 이끄는 핵심 요소이다.

Optimizer는 다음 역할을 수행한다.

1. 손실 함수의 기울기(gradient)를 계산한 뒤,
2. 파라미터를 업데이트할 방향과 크기를 정하여,
3. 점진적으로 손실 값을 줄인다.

---

# PyTorch의 Optimizer 클래스

PyTorch에서는 `torch.optim` 모듈을 통해 다양한 Optimizer 클래스들을 제공한다.  
이 클래스들은 내부적으로 경사하강법 기반의 알고리즘들을 구현하고 있으며,  
사용자는 이 Optimizer 객체를 생성하고 `.step()` 메서드를 호출하여 파라미터를 갱신할 수 있다.

PyTorch의 Optimizer는 다음과 같은 공통 인터페이스를 가진다.

```python
optimizer = torch.optim.Optimizer(model.parameters(), lr=0.01)
optimizer.zero_grad()  # 기울기 초기화
loss.backward()        # 손실함수 미분
optimizer.step()       # 파라미터 갱신
```

---

# PyTorch에서 제공하는 Optimizer 종류

다음은 PyTorch가 기본 제공하는 주요 Optimizer 클래스들이다.

- `torch.optim.SGD`  
- `torch.optim.Adam`  
- `torch.optim.AdamW`  
- `torch.optim.RMSprop`  
- `torch.optim.Adagrad`  
- `torch.optim.Adamax`  
- `torch.optim.NAdam`  

이 중 대표적으로 자주 사용되는 `SGD`, `Adam`, `AdamW`에 대해 자세히 살펴본다.

---

## 1️⃣ SGD (Stochastic Gradient Descent)

**개념**  
확률적 경사 하강법은 각 배치 또는 샘플마다 기울기를 계산해 업데이트하는 방식이다.

**수식**  

$$
\theta = \theta - \eta \cdot \nabla_\theta J(\theta)
$$

- $\theta$ : 모델 파라미터  
- $\eta$: 학습률  
- $\nabla_\theta J(\theta)$: 파라미터에 대한 손실 함수의 기울기

**특징**  
- 구현이 간단하고 빠르지만, 진동이나 수렴 불안정성이 있을 수 있다.  
- 보통 **momentum**이나 **learning rate decay**를 함께 사용하여 개선한다.

---

## 2️⃣ Adam (Adaptive Moment Estimation)

**개념**  
SGD에 모멘텀과 RMSProp 개념을 결합한 방식.  
이전 기울기의 **지수 이동 평균(1차 모멘트, 2차 모멘트)** 을 이용하여 각 파라미터별 학습률을 조정한다.

**수식 요약**

1차 모멘트:  

$$
m_t = \beta_1 \cdot m_{t-1} + (1 - \beta_1) \cdot g_t
$$

2차 모멘트:  

$$
v_t = \beta_2 \cdot v_{t-1} + (1 - \beta_2) \cdot g_t^2
$$

파라미터 업데이트:  

$$
\theta = \theta - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**특징**  
- 학습률을 자동으로 조절해 빠르게 수렴  
- 드롭아웃이나 배치 정규화 없이도 안정적인 학습 가능

---

## 3️⃣ AdamW (Adam + Weight Decay)

**개념**  
기존 Adam에 Weight Decay(가중치 감쇠)를 **정확히 적용**할 수 있도록 수정된 버전.  
Adam은 L2 정규화 적용이 일관되지 않아, **AdamW는 정규화 항을 별도로 적용**하여 해결한다.

**수식**  
Adam의 업데이트에 다음 항을 추가함:  

$$
\theta = \theta - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \cdot \theta \right)
$$

**특징**  
- Transformer 계열 모델에서 기본 옵티마이저로 사용됨  
- **일반화 성능이 더 뛰어남**



---
# 저번 주차 복습
---
# MLP의 한계와 CNN으로의 전환

### 🔹 다층 퍼셉트론(MLP)의 한계

MLP는 이론적으로 어떤 함수든 근사할 수 있지만,  
현실적으로는 다음과 같은 **구조적, 계산적 한계**를 가진다:

---
#### 1. 파라미터 폭발 (Parameter Explosion)

<img width="726" height="475" alt="image" src="https://github.com/user-attachments/assets/e3112553-31c5-423d-a171-7a01ed93262d" />

#### 🔹 예시 이미지 (16×16 입력 이미지 → MLP 구조)

- 입력 이미지: 16×16 = **256 픽셀**
- 은닉층 뉴런: 100개
- 출력층 클래스 수: 26개 (A~Z)

#### 🔹 총 파라미터 수 계산
- 입력층 → 은닉층: 256 × 100 = **25,600**
- 은닉층 바이어스: +100
- 은닉층 → 출력층: 100 × 26 = **2,600**
- 출력층 바이어스: +26

**총합 = 25,600 + 100 + 2,600 + 26 = 28,326개 파라미터**

#### 🔹 문제점
- 입력 크기가 작아도 수만 개의 파라미터 발생
- 고해상도 이미지일수록 기하급수적으로 증가
- **계산량 증가**, **학습 느림**, **과적합 위험 증가**

---
#### 2. 공간 정보 손실 (Loss of Spatial Structure)

<img width="923" height="610" alt="image" src="https://github.com/user-attachments/assets/73efc8f5-1869-46f1-8f1c-94006e90d6f0" />

#### 🔹 예시 이미지 (대문자 'A' 이미지의 위치 변화)

- 위쪽: 가운데 위치한 'A'
- 아래쪽: 왼쪽으로 2픽셀 이동한 'A'

#### 🔹 오른쪽 결과 해석
- 2픽셀 이동만으로도 **입력값 154개가 바뀜**
    - 77개는 흰색 → 검정 (0 → 1)
    - 77개는 검정 → 흰색 (1 → 0)
- MLP는 이미지를 1차원 벡터로 처리하므로,
  **공간 정보(인접성, 구조)**를 학습할 수 없음

#### 🔹 문제점
- 위치가 조금만 달라도 MLP는 완전히 다른 입력으로 인식
- 패턴이나 구조를 일반화하지 못함
- **시각적 특성에 민감** → 일반화 성능 약함

---

#### 3. 기울기 소실 문제

- 깊은 MLP에서 역전파(backpropagation) 시 기울기가 점점 작아짐
- 특히 sigmoid, tanh 같은 활성화 함수 사용 시 그 현상이 심각함
- 입력층까지 학습 신호가 도달하지 않게 되어 **학습이 멈추는 현상 발생**
- 층이 많아질수록 **학습 불가능**에 가까운 상태가 됨

→ 네트워크가 깊어질수록 성능 향상이 제한됨

---

#### 4. 일반화 성능 저하

- MLP는 모든 입력-출력을 독립적으로 학습함  
  → 파라미터 수가 많고 데이터에 과도하게 의존
- 파라미터 공유, 위치 불변성, 국소성 같은 **일반화에 유리한 구조가 없음**
- 학습 데이터에는 잘 맞지만, **새로운 데이터에는 과적합(overfitting)** 발생

---

### 🔹 CNN(Convolutional Neural Network)의 도입

CNN은 위와 같은 MLP의 한계를 해결하기 위해 고안된 구조이다.

| 문제 (MLP)       | 해결 방식 (CNN)                        |
|------------------|----------------------------------------|
| 파라미터 수 많음 | 필터를 통한 파라미터 공유             |
| 공간 정보 무시   | 국소 수용영역(Receptive Field) 유지   |
| 기울기 소실       | ReLU, BatchNorm 등 도입 가능           |
| 일반화 어려움     | 위치 불변성, 구조적 규제 내장          |

---

### 🔹 CNN 구조의 핵심 개념

- **Conv Layer**  
  입력의 일부 영역만 처리 → 지역 특성 추출 가능

- **Pooling Layer**  
  공간 크기를 줄이고, 위치 변화에 대한 강건성 확보

- **ReLU 활성화 함수**  
  비선형성과 동시에 기울기 소실 완화

- **파라미터 공유**  
  동일한 필터를 전체 입력에 반복 적용 → 학습 대상 수 감소 및 일반화 능력 향상

---
# 이번 주차 진도
---

### 1️⃣ CNN의 구조

---

### 🔹 MLP vs CNN 구조 비교

<img width="1046" height="676" alt="image" src="https://github.com/user-attachments/assets/47a679cf-b954-4d75-a766-219a57470f50" />

---

### 🔹 MLP 구조의 특징

- **두 단계로 분리된 구조**
  1. **Feature Extraction (특징 추출)**  
     사람이 수동으로 설계한 전처리 알고리즘을 사용함 
  2. **Classification (분류)**  
     수동으로 추출된 특징을 기반으로 MLP(얕은 신경망)가 분류를 수행함

- 특징 추출은 사람이 설계하고, 분류기만 학습 대상이 됨

---

### 🔹 CNN 구조의 특징

<img width="733" height="393" alt="image" src="https://github.com/user-attachments/assets/899e658d-0eef-426e-9361-4fd16f94fedd" />


- **하나의 통합 네트워크**에서 특징 추출부터 분류까지 자동으로 학습함

1. **Feature Extraction**  
   - Convolution 계층이 입력 이미지에서 특징을 자동 추출  
   - 필터는 학습을 통해 최적화됨

2. **Shift and Distortion Invariance**  
   - Pooling 계층이 위치 변화나 왜곡에도 강인한 표현을 생성

3. **Classification**  
   - 추출된 특징을 기반으로 Fully Connected Layer(MLP)가 최종 분류를 수행

- 전체 네트워크가 깊은 신경망(Deep NN) 구조로 구성되며,  
  특징 추출과 분류를 **통합적으로 학습**함

---

### 2️⃣ CNN 구조의 핵심 요소 – Convolution 연산의 원리

---

<img width="814" height="319" alt="image" src="https://github.com/user-attachments/assets/5f042013-500d-4910-a93a-b4d022cec58a" />

### 🔹 필터(커널)의 개념

- **필터(filter) 또는 커널(kernel)**은 작은 크기의 행렬이다 (예: 3×3, 5×5)
- 이 필터를 **입력 이미지 위에서 슬라이딩**하면서, 해당 위치의 값들과 필터의 값을 곱한 뒤 더한다
- 이 연산을 통해 하나의 **출력 값**(feature map의 한 픽셀)을 생성함

---

### 🔹 국소 영역(local receptive field)

- Convolution은 **전체 이미지가 아닌 일부분만을 보는 방식**
- 예: 32×32 이미지에 5×5 필터 → 5×5 구역씩만 보면서 특징을 추출
- 이 방식은 입력의 **공간적 구조(locality)**를 보존하며,
  **위치 변화에 강건한 표현**을 만들 수 있게 해준다

---

### 🔹 스트라이드(Strides)

- 필터가 **한 번에 몇 칸씩 이동**하는지를 정하는 값
- 기본값은 1 (한 칸씩 이동), 스트라이드가 커질수록 출력 크기는 작아짐
- Downsampling 효과가 있음

---

### 🔹 패딩(Padding)

<img width="278" height="277" alt="image" src="https://github.com/user-attachments/assets/5404320e-a67b-4a78-90c3-a6121ac819b2" />

- 필터가 가장자리를 완전히 커버하지 못하는 문제를 해결하기 위해,
  입력의 테두리에 **0을 덧붙이는 것**
- **Same padding**: 출력 크기를 입력과 동일하게 유지
- **Valid padding**: 패딩 없이 계산 → 출력 크기가 작아짐

---
### 🔹 Convolution 설정에 따른 출력 크기 비교

다음은 입력 크기 5×5, 필터 크기 3×3, 필터 수 1개를 기준으로  
세 가지 경우에 대한 출력 결과만 정리한 것이다.

<img width="833" height="328" alt="image" src="https://github.com/user-attachments/assets/59852128-9ae7-4da9-85e9-19e8c40e382d" />

---

### 🔹 1. 스트라이드 1, 패딩 없음 (No Padding)

- 입력 크기: 4×4  
- 필터 크기: 3×3  
- 패딩: 0  
- 스트라이드: 1  

**→ 출력 크기 = 2×2**

---

### 🔹 2. 스트라이드 1, 패딩 2

- 입력 크기: 5×5  
- 필터 크기: 4×4  
- 패딩: 2  
- 스트라이드: 1  

**→ 출력 크기 = 6×6**

---

### 🔹 3. 스트라이드 2, 패딩 1

- 입력 크기: 6×6  
- 필터 크기: 3×3  
- 패딩: 1  
- 스트라이드: 2  

**→ 출력 크기 = 3×3**


---

### 🔹 출력 Feature Map 크기 계산 공식

- 입력 크기: W × H  
- 필터 크기: F  
- 패딩: P  
- 스트라이드: S

계산식:

$$
\text{Output Size} = \left\lfloor \frac{W - F + 2P}{S} \right\rfloor + 1
$$

---
