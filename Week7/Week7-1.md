# 저번 수업 복습

---
## 1️⃣ RNN

---
### 🔹 기존 신경망의 한계

완전연결 신경망(Dense Layer)을 이용해도 각 시점의 데이터를 독립적으로 처리할 수는 있다.

하지만 다음과 같은 문제점이 발생한다.

#### ❌ 시간 순서가 고려되지 않음
입력 데이터를 한 번에 flatten하거나, 같은 네트워크로 처리하면  
각 시점 사이의 연관성이나 순서 정보가 사라진다.

#### ❌ 과거 정보를 기억하지 못함
이전 시점의 정보가 현재의 예측에 반영되어야 하는 경우,  
기존 신경망은 매 시점의 입력만을 기반으로 예측하므로 **문맥이나 맥락이 반영되지 않는다.**

---
### 🔹 RNN의 구조 및 상태(State)

RNN에서 **state**는 일종의 **기억(Memory)** 을 의미하며,  
입력 데이터를 시간 순서에 따라 처리하면서 **과거 정보를 저장하고 다음 시점에 전달**한다.

RNN은 일반적으로 다음의 **세 가지 상태**로 구성된다:

| 상태 종류                | 기호    | 역할                                               |
| -------------------- | ----- | ------------------------------------------------ |
| 입력 상태 (Input State)  | $x_t$ | 시점 $t$에서 들어오는 입력값                                |
| 은닉 상태 (Hidden State) | $h_t$ | 이전 시점의 은닉 상태 $h_{t-1}$와 입력 $x_t$를 바탕으로 계산된 기억 정보 |
| 출력 상태 (Output State) | $y_t$ | 은닉 상태를 바탕으로 계산된 출력값 (예: 예측 결과)                   |

$h_t = f(x_t, h_{t-1})$  
$y_t = g(h_t)$

이처럼 **입력, 은닉, 출력 상태**는 시간 축을 따라 반복적으로 연결되어  
RNN이 **시간적 의존성**을 학습할 수 있도록 돕는다.

<img width="610" height="236" alt="image" src="https://github.com/user-attachments/assets/bf574b33-9569-4a3b-9aa6-63118eed0256" />

RNN은 히든 노드가 방향을 가진 엣지로 연결되 순환구조를 이루는 인공신경망의 한 종류로, 구조는 위 사진과 같이 나뉜다.

- 기본 구조
  - 1. one to many : 여러 시점 X를 통한 하나의 Y를 예측
  	- 예) 여러 시점에서 데이터를 통해 특정 시점의 제품 상태 예측 
  - 2. many to one : 단일 시점 X로 순차적인 Y를 예측
	- 예) 이미지 데이터 -> 이미지 데이터에 대한 문장 출력
  - 3. many to many : 순차적인 X -> 순차적인 Y 예측
	- 예) 문장 -> 각 단어의 품사 예측

---
## 2️⃣ RNN의 기본 구조: 순환 연결과 Hidden State


---
### 🔹 순환(Recurrent) 구조란?

RNN의 핵심은 **출력값을 다시 자기 자신에게 입력으로 넣는**  
**순환 구조(recurrent connection)** 에 있다.

이 구조는 시간에 따라 변화하는 데이터를 처리하면서,  
과거의 정보를 현재까지 **연결해서 기억**할 수 있게 해준다.

---
### 🔹 시각화: 시간축으로 펼친 구조 (Unrolling)

RNN의 순환 구조는 시간에 따라 **반복적으로 연결**되는 형태이다.  
이를 **시간축으로 펼치면** 아래 그림처럼 **각 시점마다 동일한 연산이 반복되는 구조**로 표현할 수 있다.

<img width="845" height="330" alt="image" src="https://github.com/user-attachments/assets/ac7bfe12-b5bb-4798-9e9b-c261b368069d" />



- 모든 시점에서 **같은 가중치**(예: $W_{xh}, W_{hh}, W_{hy}$)를 공유하여 학습
- **이전 시점의 은닉 상태** $h_{t-1}$ 이 현재 시점의 은닉 상태 $h_t$ 계산에 영향을 줌
- 마지막 시점의 은닉 상태 $h_t$를 통해 최종 출력 $y_t$ 생성

이러한 구조 덕분에 RNN은 **시간적인 연속성**과 **문맥 정보**를 반영할 수 있다.


---
#### 🔹 Unrolling의 목적

- 시간적 흐름을 이해하기 쉽도록 시각화
- BPTT(시간 역전파)를 설명하기 위해 필수적인 개념
- 시퀀스 전체에 걸친 **의존 관계**를 추적할 수 있음
  
---
### 🔹 기본적인 RNN 셀의 수식 구조

<img width="649" height="380" alt="image" src="https://github.com/user-attachments/assets/bc7bce28-d8aa-4d41-8d23-1cd0a2efd6d5" />

하나의 RNN 셀(Cell)은 아래와 같은 계산을 수행한다.

$$
h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)
$$

이 수식은 **과거의 정보($h_{t-1}$)** 와 **현재 입력($x_t$)** 을  
선형 결합한 후, $\tanh$ 함수를 통과시켜 **현재의 상태($h_t$)** 를 구하는 방식이다.

즉, RNN은 **과거와 현재의 정보를 합성**하여 다음 상태를 만들어낸다.


$$
y_t = W_{hy}h_t + b_y
$$

이 수식은 은닉 상태 $h_t$를 선형 변환하여 출력 $y_t$를 구한다.  
분류 문제의 경우, 출력에 소프트맥스 함수를 적용하여 확률 분포로 변환할 수 있다.


- $x_t$: 시점 $t$의 입력  
- $h_{t-1}$: 이전 시점의 은닉 상태 (메모리 역할)  
- $h_t$: 현재 시점의 은닉 상태 (업데이트된 메모리)  
- $y_t$: 현재 시점의 출력  
- $\tanh$: 비선형 활성화 함수 (또는 ReLU 등)

---
### 🔹 Hidden State의 역할

- $h_t$는 이전 상태와 현재 입력 정보를 반영하여 계산된 **기억 상태**이다.
- 이 값이 다음 시점으로 전달되며, **시계열 정보를 압축하여 보존**한다.
- 결과적으로 RNN은 입력의 순서를 인식하고, 그에 따라 **동적으로 반응**할 수 있다.

---

### 🔹 은닉 상태의 시간 축 흐름

$$
h_0 \rightarrow h_1 \rightarrow h_2 \rightarrow \cdots \rightarrow h_t
$$

각 시점의 $h_t$는 이전 시점의 $h_{t-1}$에 의존하므로,  
**RNN은 과거 정보를 누적하여 처리하는 구조**를 갖는다.

이 구조 덕분에 RNN은 다음과 같은 문제를 다룰 수 있다.

- 문장의 문맥 파악
- 시계열 예측
- 연속적인 신호 처리 등

---

### 🔹 계산 흐름

입력 (xₜ) + 이전 상태 (hₜ₋₁)  
	↓  
선형 변환 + tanh  
	↓  
현재 상태 (hₜ)  
	↓  
선형 변환  
	↓  
출력 (yₜ)

---

## 4️⃣ BPTT (Backpropagation Through Time)

---

### 🔹 BPTT

BPTT는 RNN에서 시간 축을 따라 펼쳐진 모든 시점의 손실을 합산하여,
전체 가중치에 대해 역전파를 수행하는 학습 방법이이다.

---
### 🔹 주요 특징

1. **가중치 공유**  
2. **시간 축을 따라 기울기가 누적**  


---
### 🔹 시간 축을 펼친 구조에서의 역전파

RNN을 시점 $t=1$부터 $t=T$까지 펼친 후,  
출력 오차를 기준으로 **맨 끝 시점부터 역방향으로 기울기를 전파**한다.

전체 시퀀스의 손실은 다음과 같다. 

$$
L = \sum_{t=1}^T \mathcal{L}(y_t, \hat{y}_t)
$$

- $L$: 전체 손실 함수 (전체 시퀀스에 대한 합)  
- $y_t$: 예측값  
- $\hat{y}_t$: 실제 정답  
- $\mathcal{L}$: 시점별 손실 함수 (예: 크로스엔트로피)

이때 가중치 $W_{hh}$에 대해 기울기를 구하려면, 모든 시간의 은닉 상태 $h_t$에 대해 미분해야 한다.

$$
\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}
$$

여기서  $\frac{\partial h_t}{\partial W_{hh}}$ 계산 시 $h_{t-1}, h_{t-2}, \dots$ 까지의 정보가 모두 연결되어 있다.


---

### 🔹 시각적 설명


many - to - one 의 경우 그림과 같이 역전파가 계산된다.

<img width="877" height="373" alt="image" src="https://github.com/user-attachments/assets/5edeb4cf-4ea8-48b3-a84a-fec981a97fe9" />

### 출력층 $W_{hy}$에 대한 손실의 미분

$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial W}
$$

$z = W_{hy} h_t$: 출력층에서 은닉 상태 $h_t$에 선형 변환을 적용한 값 

---

```math
\frac{\partial \text{Loss}}{\partial W_{hh}} = 
\underbrace{
\frac{\partial \text{Loss}}{\partial \hat{y}_3} 
\cdot \frac{\partial \hat{y}_3}{\partial h_3} 
\cdot \frac{\partial h_3}{\partial W_{hh}}
}_{\tau_3: 시점 3에서의 영향}
\,+\,
\underbrace{
\frac{\partial \text{Loss}}{\partial \hat{y}_3} 
\cdot \frac{\partial \hat{y}_3}{\partial h_3} 
\cdot \frac{\partial h_3}{\partial h_2} 
\cdot \frac{\partial h_2}{\partial W_{hh}}
}_{\tau_2: 시점 2로부터 전해진 영향}
\,+\,
\underbrace{
\frac{\partial \text{Loss}}{\partial \hat{y}_3} 
\cdot \frac{\partial \hat{y}_3}{\partial h_3} 
\cdot \frac{\partial h_3}{\partial h_2} 
\cdot \frac{\partial h_2}{\partial h_1} 
\cdot \frac{\partial h_1}{\partial W_{hh}}
}_{\tau_1: 시점 1로부터 전해진 영향}
```

---
### 🔹 단어 예측 예제

입력된 문장을 보고 다음 단어를 예측하는 RNN 모델을 생각해보자.

예시 문장 : 나는 → 밥을 → 먹었 → 다

- **입력 시퀀스 (X)**: 나는, 밥을, 먹었  
- **정답 시퀀스 (Ŷ)**: 밥을, 먹었, 다

RNN 처리 흐름 예시

| 시점 | 입력 ($x_t$) | 정답 ($\hat{y}_t$) | 출력 ($y_t$) | 손실 $\mathcal{L}(y_t, \hat{y}_t)$ |
|------|--------------|---------------------|--------------|----------------------------|
| t=1  | 나는         | 밥을                | 먹었         | 손실₁ = 1.2                |
| t=2  | 밥을         | 먹었                | 먹었         | 손실₂ = 0.3                |
| t=3  | 먹었         | 다                  | 좋아         | 손실₃ = 2.1                |

모든 시점에서의 손실을 더해서 전체 손실 $L$을 계산:

$$
L = \text{손실}_1 + \text{손실}_2 + \text{손실}_3 = 1.2 + 0.3 + 2.1 = 3.6
$$

RNN 학습에서는

- 각 시점마다 따로 가중치를 업데이트하지 않음
- **모든 손실을 더한 총합 손실 $L$에 대해 한 번에 역전파** 수행

즉, 전체 손실 L = 3.6 에 대한 ∂L/∂W 를 계산 → 모든 가중치 한 번에 업데이트한다.

---
## 5️⃣ RNN의 한계

---
### 🔹 장기 의존성 문제 (Long-Term Dependency)

RNN은 시간에 따른 정보를 처리할 수 있는 구조를 가지고 있지만,  
**멀리 떨어진 과거 정보는 현재에 잘 전달되지 않는 한계**를 갖고 있다.

이 문제를 **장기 의존성 문제 (long-term dependency)** 라고 부른다.

Sequence의 길이가 길어질수록, 과거 정보 학습에 어려움이 발생한다.
원인 : 기울기 소실 및 기울기 폭주

BPTT를 통해 가중치에 대한 기울기를 계산할 때,  
다수의 연쇄된 곱셈으로 인해 기울기가 아래처럼 표현된다.

$$
\frac{\partial L}{\partial W_{hh}} \propto \prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}}
$$

은닉 상태는 아래와 같이 활성화 함수를 포함한 선형 결합이다.

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

그 결과, 역전파에서 기울기 소실 및 기울기 폭주 문제가 발생한다.


---
# 이번 주차 수업
---

### 🔸 기존 RNN 구조 요약

- 입력 $x_t$와 이전 hidden state $h_{t-1}$을 이용하여 현재 hidden state $h_t$를 계산
- 반복적인 구조로 시퀀스 데이터를 처리함

#### 수식:
- $h_{t-1} = f(W_{xh} x_{t-1} + W_{hh} h_{t-2})$
- $h_t = f(W_{xh} x_t + W_{hh} h_{t-1})$
- $y_t = g(W_{hy} h_t)$

---

<img width="1153" height="524" alt="image" src="https://github.com/user-attachments/assets/b2db4e2c-bee2-47f6-a078-e28b60d055be" />

---

### 🔹 순환신경망 개선 모델 : Long Short-Term Memory (LSTM)

- LSTM은 장기 의존성 문제를 완화한 RNN 개선 모델
- Cell state ($C_t$) 구조를 제안하고, 세 가지 gate를 추가한 구조
  - 세 가지 gate: Forget gate ($f_t$), Input gate ($i_t$), Output gate ($o_t$)
  - 각 게이트는 시그모이드 함수를 통과하므 0과 1사이의 간단한 벡터로 이해

### 🔸 LSTM 구조 수식

- $f_t = \sigma(W_{xf} x_t + W_{hf} h_{t-1} + b_f)$  // Forget gate
- $i_t = \sigma(W_{xi} x_t + W_{hi} h_{t-1} + b_i)$  // Input gate
- $o_t = \sigma(W_{xo} x_t + W_{ho} h_{t-1} + b_o)$  // Output gate
- $\tilde{C}_t = \tanh(W_{xg} x_t + W_{hg} h_{t-1} + b_g)$  // 후보 cell state
- $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$  // 새로운 cell state
- $h_t = o_t \odot \tanh(C_t)$  // 최종 hidden state
  
---

### 🔹 LSTM 구조 상세도

LSTM은 입력 $x_t$, 이전 hidden state $h_{t-1}$, 이전 cell state $C_{t-1}$을 받아  
현재 cell state $C_t$와 hidden state $h_t$를 계산한다.

<img width="1017" height="553" alt="image" src="https://github.com/user-attachments/assets/a5a473ed-d213-4793-9310-8b38827edfac" />

---

### 🔸 셀 상태 및 은닉 상태 계산

- 임시 셀 상태: $\tilde{C}_t$
- $= \tanh(W_{xh\_g} x_t + W_{hh\_g} h_{t-1} + b_{h\_g})$
- 현재 셀 상태: $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
- 현재 은닉 상태: $h_t = o_t \odot \tanh(C_t)$

---

### 🔸 구성 요소 설명

- $\sigma$: sigmoid 함수 (0~1 범위로 조절)
- $\tanh$: tanh 함수 (값 스케일링 및 후보 생성)
- $\odot$: 요소별 곱(Element-wise multiplication)
- $+$: 셀 상태 누적 (기억 유지)

---

### 🔹 LSTM 연산 과정 요약

<img width="807" height="624" alt="image" src="https://github.com/user-attachments/assets/f374be55-e642-4119-835c-1acab1825b0d" />

LSTM은 입력 $x_t$, 이전 상태 $h_{t-1}$, $C_{t-1}$을 기반으로 다음과 같은 순서로 계산된다:

---

### 🔹 순전파 과정

##### 1️⃣ Gate 계산하기: Forget ($f_t$), Input ($i_t$), Output ($o_t$)

- $f_t = \sigma(W_{xh_f} x_t + W_{hh_f} h_{t-1} + b_{h_f})$
- $i_t = \sigma(W_{xh_i} x_t + W_{hh_i} h_{t-1} + b_{h_i})$
- $o_t = \sigma(W_{xh_o} x_t + W_{hh_o} h_{t-1} + b_{h_o})$


##### 2️⃣ Cell state ($C_t$) 업데이트 하기

<img width="1245" height="164" alt="image" src="https://github.com/user-attachments/assets/ed55acc0-34e3-4812-b997-d0a2faae7892" />

- 후보 셀 상태:  
  -  $\tilde{C}_t$
  -  $= \tanh(W_{xh_g} x_t + W_{hh_g} h_{t-1} + b_{h_g})$

- 현재 셀 상태 계산:  
  $C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$
  
- forget gate의 의미 : 과거의 정보를 컨트롤 하는 역할
- input gate의 의미 : 현재의 정보를 컨트롤 하는 역할
- 0을 주면 값을 지우고, 0보다 크며 1 보다 같거나 작은 값을 주면 정보를 기억하는 역할


##### 3️⃣ Hidden state ($h_t$) 업데이트 하기

- $h_t = o_t \odot \tanh(C_t)$

---

### 🔹 손실 함수 계산

- 마지막 출력 $h_t$ 또는 시퀀스 전체 출력 ${h_1, h_2, \dots}$를 기반으로 예측값 $y_t$ 생성
- 예측값과 정답 $\hat{y}_t$ 간의 오차(loss) 를 계산
  - 보통 Cross Entropy 또는 MSE 사용
 
---

### 🔹 역전파 (BPTT through time)

LSTM도 RNN처럼 시간 축을 따라 펼친 후 BPTT (Backpropagation Through Time) 으로 학습한다.

- 시간 순서대로 손실 기울기 $\frac{\partial L}{\partial h_t}$ 를 계산
- 각 게이트와 셀 가중치에 대해 미분 수행
- 모든 시점의 기울기를 누적하여 가중치 파라미터들을 업데이트

즉, LSTM은 단순히 하나의 층처럼 학습되지 않고,
게이트들의 출력에 대한 미분이 연결된 구조로 역전파가 이뤄진.

---

### 🔹 옵티마이저를 통한 업데이트

- 최종적으로, 계산된 기울기들은 옵티마이저(예: SGD, Adam)를 통해 다음과 같이 파라미터를 업데이트
  - W ← W - η × ∂L/∂W
 
- 위 과정을 반복

---

### 🔹 GRU란?

<img width="563" height="361" alt="image" src="https://github.com/user-attachments/assets/30d968cd-8901-4cb0-aff0-419baccb2124" />

- GRU는 LSTM보다 단순하고 계산 효율이 높은 RNN 변형 구조이다.
- 게이트 구조를 통해 RNN의 기울기 소실 문제를 완화하면서도, LSTM보다 구조가 간단하다.
- LSTM의 3개 게이트(입력, 삭제, 출력) 대신 2개의 게이트만 사용
- 셀 상태 없이 은닉 상태 $h_t$ 하나만 유지
- 필요한 정보를 선택적으로 기억/갱신/초기화하는 방식

---

---

### 🔹 GRU의 구성 요소

<img width="1137" height="332" alt="image" src="https://github.com/user-attachments/assets/a3790d40-a95d-4675-9775-b72096682b30" />


| 구성 요소 | 설명 |
|-----------|------|
| 업데이트 게이트 ($z_t$) | 과거 정보를 얼마나 유지할지 결정 |
| 리셋 게이트 ($r_t$) | 과거 정보를 얼마나 초기화할지 결정 |
| 후보 은닉 상태 ($\tilde{h}_t$) | 현재 입력과 과거 정보를 바탕으로 계산되는 새로운 정보 |
| 최종 은닉 상태 ($h_t$) | 이전 은닉 상태와 후보 은닉 상태의 가중 평균 |

- Output gate 제거 및 reset, update gate 사용

  -  **Reset gate**  
   $r_t = \sigma(W_{xr} x_t + W_{hr} h_{t-1} + b_{hr})$

  - **Update gate**  
    $z_t = \sigma(W_{xz} x_t + W_{hz} h_{t-1} + b_{hz})$

  - **임시 cell state**  
    - $\tilde{c}_t$
    - $= \tanh(W_{xg} x_t + W_{hg} (r_t \odot h_{t-1}) + b_{hg})$

- Cell state와 Hidden state 통합


  - 최종 hidden state (= cell state)

    - $h_t = (1 - z_t) \odot h_{t-1}$
    - $+ z_t \odot \tilde{c}_t$

        - $(1 - z_t)$: Forget gate 역할  
        - $z_t$: Input gate 역할

---

### 🔹 정리

시계열 데이터는 순서가 중요한 데이터를 의미하며, 이러한 특성에 특화된 모델로는 RNN, LSTM, GRU가 있다.
여기서 "순서가 있다"는 의미는 현재의 정보를 예측하거나 이해하는 데 과거의 정보가 도움이 될 수 있다는 것을 뜻한다.

이 세 가지 모델은 모두 과거의 정보를 어떻게 활용할 것인가에 따라 구조적으로 차이를 보인다.
공통적으로 과거의 정보를 hidden state에 담아 전달하며, 이전 시점의 hidden state를 현재 시점의 hidden state 계산에 사용한다.

즉, 각 시점의 hidden vector $h_t$ 가 중요한 역할을 하며, 이를 계산하는 방식에 차이가 존재한다:

- RNN은 단순히 **이전 hidden state $h_{t-1}$과 현재 입력 $x_t$**를 결합하여 $h_t$를 계산한다.

- LSTM은 **게이트 구조(gate mechanism)**를 도입하여,
  과거 정보를 얼마나 기억할지(forget gate), **새로운 정보를 얼마나 반영할지(input gate)**를 학습된 가중치로 조절한다.

- GRU는 LSTM보다 간단한 구조로, update gate 하나로 forget과 input 기능을 통합하고,
  reset gate를 통해 과거 정보를 얼마나 초기화할지를 결정한다.

결론적으로, 세 모델 모두 이전 hidden state를 현재 시점에 반영하지만,
그 반영 방식을 단순 연결(RNN), **게이트 기반의 조절(LSTM, GRU)**로 확장하며 더 정교하게 정보를 조절한다는 차이가 있다.
