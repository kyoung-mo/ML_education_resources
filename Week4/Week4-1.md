# 📘 Week4‑1 – 손실 함수 & 역전파

---

## 📉 손실 함수 *(Loss Function)*

### 🔍 `y`·`ŷ`(y‑hat) 기호 이해
| 기호 | 읽는 법 | 의미 |
|------|--------|------|
| **`y`** | 와이 | **실제값**·정답(Ground Truth) |
| **`ŷ`** (y‑hat) | 와이‑햇 | **모델이 예측한 값**·추정치(Prediction) |

> 모자(^) 모양 **“hat”** 은 통계·머신러닝에서 “추정(estimation)”을 나타내는 관례적 표기입니다.  
> 즉, 손실 함수는 **실제값 `y`** 와 **예측값 `ŷ`** 의 차이를 수치화합니다.

---

### 1️⃣ 손실 함수 모음

| 범주 | 함수 | 식 (요약) | 특징 / 사용처 |
|------|------|-----------|----------------|
| 회귀 | **MSE** | $\dfrac{1}{N} \sum (y - \hat{y})^2$ | 이상치 민감, 보편적 |
|      | MAE | $\dfrac{1}{N} \sum \lvert y - \hat{y} \rvert$ | 이상치 강인 |
| 이진 분류 | **BCE** | $- \big[ y \log \hat{y} + (1 - y) \log (1 - \hat{y}) \big]$ | 시그모이드 이후 사용 |
| 다중 분류 | **Cross-Entropy** | $- \sum y_k \log \hat{p}_k$ | 소프트맥스 이후 |
| 불균형 | Focal Loss | $- (1 - \hat{p})^{\gamma} \log \hat{p}$ | 어려운 샘플 집중 |
| 분포 거리 | KL Divergence | $\sum p \log \frac{p}{q}$ | 지식 증류, VAE |

> 여기서 $\hat{p}$ 역시 **예측 확률**을 뜻합니다.

---

### 2️⃣ PyTorch 예제

```python
import torch
import torch.nn as nn

logits  = torch.tensor([[2.0, 0.5, -1.0]])  # 모델이 낸 점수(logit)
targets = torch.tensor([0])                 # 정답 레이블

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, targets)
print("Cross‑Entropy Loss:", loss.item())
```

---

## 🔁 역전파 *(Backpropagation)*

### 1️⃣ 개념
- 손실이 줄어들도록 **파라미터(가중치·편향)** 를 업데이트하는 알고리즘  
- **체인 룰**로 다층 미분을 효율적으로 계산

### 2️⃣ PyTorch 학습 절차

| 단계 | 코드 | 역할 |
|------|------|------|
| 순전파 | `y_pred = model(x)` | 그래프 생성 |
| 손실 계산 | `loss = criterion(y_pred, y)` | 노드 추가 |
| 역전파 | `loss.backward()` | ∂L/∂θ 계산 |
| 업데이트 | `optimizer.step()` | θ ← θ − η·∂L/∂θ |
| 초기화 | `optimizer.zero_grad()` | 그래디언트 리셋 |

---

## 🔬 [실습] 넘파이 vs 파이토치로 2단 MLP “역전파(gradient)” 비교

### 1️⃣ 넘파이로 (수동 미분)

```python
import numpy as np

def sigmoid(x): return 1 / (1 + np.exp(-x))
def d_sigmoid(x): return sigmoid(x) * (1 - sigmoid(x))

x = np.array([1.0, 0.5])
y_true = np.array([1.0])

W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
b1 = np.array([0.1, 0.2])
W2 = np.array([[0.5], [0.6]])
b2 = np.array([0.3])

z1 = np.dot(x, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
a2 = sigmoid(z2)
loss = 0.5 * np.sum((a2 - y_true)**2)

d_loss_a2 = a2 - y_true
d_loss_z2 = d_loss_a2 * d_sigmoid(z2)
d_loss_W2 = np.outer(a1, d_loss_z2)
d_loss_b2 = d_loss_z2

d_loss_a1 = np.dot(W2, d_loss_z2)
d_loss_z1 = d_loss_a1 * d_sigmoid(z1)
d_loss_W1 = np.outer(x, d_loss_z1)
d_loss_b1 = d_loss_z1

print("NumPy dW1:", d_loss_W1)
print("NumPy dW2:", d_loss_W2)
```

### 🧪 수치 미분으로 검증

```python
def numerical_gradient(f, W, h=1e-5):
    grad = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            tmp = W[i, j]
            W[i, j] = tmp + h
            fxh1 = f()
            W[i, j] = tmp
            fx = f()
            grad[i, j] = (fxh1 - fx) / h
    return grad

def loss_func_w1():
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return 0.5 * np.sum((a2 - y_true)**2)

num_grad_W1 = numerical_gradient(loss_func_w1, W1)
print("수치 미분 dW1:", num_grad_W1)
print("해석 미분 dW1:", d_loss_W1)
```

---

### 2️⃣ 파이토치로 (자동 미분)

```python
import torch

W1 = torch.tensor([[0.1, 0.2], [0.3, 0.4]], requires_grad=True)
b1 = torch.tensor([0.1, 0.2], requires_grad=True)
W2 = torch.tensor([[0.5], [0.6]], requires_grad=True)
b2 = torch.tensor([0.3], requires_grad=True)
x = torch.tensor([1.0, 0.5])
y_true = torch.tensor([1.0])

def sigmoid(x): return 1 / (1 + torch.exp(-x))

z1 = torch.matmul(x, W1) + b1
a1 = sigmoid(z1)
z2 = torch.matmul(a1, W2) + b2
a2 = sigmoid(z2)
loss = 0.5 * ((a2 - y_true) ** 2).sum()
loss.backward()

print("PyTorch dW1:", W1.grad)
print("PyTorch dW2:", W2.grad)
```

---

## 🛠️ 실습: 선형 회귀로 `y=3x` 근사

```python
import torch, torch.nn as nn, torch.optim as optim

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

x = torch.tensor([[1.0], [2.0], [3.0]])
y = torch.tensor([[3.0], [6.0], [9.0]])

for epoch in range(200):
    pred = model(x)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}: Loss={loss.item():.4f}")

print("학습된 w, b:", model.weight.item(), model.bias.item())
```

---

## 🎯 과제

1. 위 예제를 참고해 `nn.Linear(1,1)` 로 **`y = 3x`** 근사하기  
2. `nn.MSELoss()` 사용, **에폭마다 손실** 출력  
3. 학습 종료 후 **weight(≈3) & bias(≈0)** 확인  
4. 2단 MLP 구조로 위와 같은 넘파이/파이토치 역전파 실습 직접 수행  
5. 추가 실험: **수치 미분으로 해석 미분 결과 검증**

---

✅ **환경**: Python 3.x, Google Colab, numpy, PyTorch ≥ 2.0  
설치: `!pip install numpy torch torchvision -q`
