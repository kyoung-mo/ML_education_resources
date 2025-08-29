
---
# 저번 수업 복습
---

### 🔸 기존 RNN 구조 요약

- 입력 $x_t$와 이전 hidden state $h_{t-1}$을 이용하여 현재 hidden state $h_t$를 계산
- 반복적인 구조로 시퀀스 데이터를 처리함

#### 수식:
- $h_{t-1} = f(W_{xh} x_{t-1} + W_{hh} h_{t-2})$
- $h_t = f(W_{xh} x_t + W_{hh} h_{t-1})$
- $y_t = g(W_{hy} h_t)$

---

<img width="1153" height="524" alt="image" src="https://github.com/user-attachments/assets/177a4e7b-bad1-4b22-add1-c5ae432eb5d5" />


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

<img width="1017" height="553" alt="image" src="https://github.com/user-attachments/assets/4699f178-08a3-46d5-bcff-4b18f128d8ec" />


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

<img width="807" height="624" alt="image" src="https://github.com/user-attachments/assets/3beeb615-7b6b-44c2-8bdf-09a8d0654eb8" />


LSTM은 입력 $x_t$, 이전 상태 $h_{t-1}$, $C_{t-1}$을 기반으로 다음과 같은 순서로 계산된다:

---

### 🔹 순전파 과정

##### 1️⃣ Gate 계산하기: Forget ($f_t$), Input ($i_t$), Output ($o_t$)

- $f_t = \sigma(W_{xh_f} x_t + W_{hh_f} h_{t-1} + b_{h_f})$
- $i_t = \sigma(W_{xh_i} x_t + W_{hh_i} h_{t-1} + b_{h_i})$
- $o_t = \sigma(W_{xh_o} x_t + W_{hh_o} h_{t-1} + b_{h_o})$


##### 2️⃣ Cell state ($C_t$) 업데이트 하기

<img width="1245" height="164" alt="image" src="https://github.com/user-attachments/assets/7742a503-d6c9-46e9-af92-e9641c0c68ae" />


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

<img width="563" height="361" alt="image" src="https://github.com/user-attachments/assets/df9f3e77-cbca-4e7b-bbd8-c3586c39ae0c" />


- GRU는 LSTM보다 단순하고 계산 효율이 높은 RNN 변형 구조이다.
- 게이트 구조를 통해 RNN의 기울기 소실 문제를 완화하면서도, LSTM보다 구조가 간단하다.
- LSTM의 3개 게이트(입력, 삭제, 출력) 대신 2개의 게이트만 사용
- 셀 상태 없이 은닉 상태 $h_t$ 하나만 유지
- 필요한 정보를 선택적으로 기억/갱신/초기화하는 방식

---

### 🔹 GRU의 구성 요소

<img width="1137" height="332" alt="image" src="https://github.com/user-attachments/assets/e35e162e-34b4-477a-9d3e-dc3a8c288e45" />



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

---
# 이번 주차 수업
---

<img width="1216" height="476" alt="image" src="https://github.com/user-attachments/assets/c2c244c1-ce14-4c5b-8bd6-f76f2dcdec74" />


### 🔹 Attention is All You Need (2017, NeurIPS)  
- 2024년 11월 기준 **140,000여 회 인용**  
- 시계열 데이터에 적합하도록 제안된 알고리즘으로 **RNN 기반 방법론을 대체**  
- 후속 연구에서 다양한 관점의 **모델 구조 변형 및 응용 연구**에 활용  

### 🔹 주요 참고 자료  
- **Transformer 원논문**: *Attention Is All You Need* (Vaswani et al., 2017)  
- **Survey 논문**: Transformer 구조와 응용을 종합적으로 다룬 리뷰 논문  
- **시각 자료**: Jay Alammar의 Transformer 시각화 자료 (YouTube, 블로그)  

---

<img width="1305" height="554" alt="image" src="https://github.com/user-attachments/assets/8b160d4e-4024-4fcd-8c2d-3008b4b34d0f" />


### 🔹 Attention의 개념  
- **Attention** = 인코더와 디코더 **사이**의 가중치 (weight between)  
- 번역 등에서 입력 문장의 각 단어가 출력 문장의 특정 단어와 얼마나 관련 있는지 계산  
- 예: 영어 "I love you" → 한국어 "나는 너를 사랑해" 변환 시, 각 단어의 연관도 계산 후 반영

### 🔹 Self-Attention의 개념  
- **Self-Attention** = 인코더 **내부** 또는 디코더 **내부**의 가중치 (weight within)  
- 한 문장 내에서 단어들끼리 서로 얼마나 관련 있는지 계산  
- 각 단어가 문장의 다른 단어들과 상호작용해 문맥 정보 학습 가능  
- 예: "나는 학생이다"에서 **"학생"**은 "나는"과 높은 연관도를, "이다"와 낮은 연관도를 가질 수 있음

---

<img width="864" height="549" alt="image" src="https://github.com/user-attachments/assets/bcfd85e6-2509-4d81-9b8b-673e63f2eea3" />


### 🔹 Transformer의 기본 구조: Encoder-Decoder  
- Transformer는 **인코더(Encoder)**와 **디코더(Decoder)**로 구성  
- **Encoder**: 입력 문장을 받아 각 단어의 문맥(Context) 벡터 생성  
- **Decoder**: 인코더에서 전달받은 문맥 벡터와 이전에 생성된 단어를 바탕으로 다음 단어 예측  

### 🔹 동작 예시  
- **입력(Inputs)**: "나는 학생이다"  
- **출력(Output)**: "I am a student"  
- 인코더가 한국어 문장을 벡터로 변환 → 디코더가 이를 영어 문장으로 변환  

### 🔹 특징  
- Encoder-Decoder 구조 덕분에 입력과 출력의 길이가 달라도 처리 가능  
- 번역, 요약, 질의응답 등 다양한 시퀀스 변환(Sequence-to-Sequence) 작업에 활용  

---

<img width="845" height="531" alt="image" src="https://github.com/user-attachments/assets/2d2e3400-15eb-4e64-b0be-82ec9b0f6f93" />


### 🔹 Transformer의 Encoder-Decoder 다층 구조  
- 인코더와 디코더는 각각 **N개의 모듈**로 구성되며, 모든 모듈의 **구조는 동일**  
- 각 모듈의 **파라미터 값은 서로 다름**  
- 이전 모듈의 **출력(Output)** 이 다음 모듈의 **입력(Input)** 으로 전달됨  
  - 단, 첫 번째 모듈의 입력은 원래 입력 시퀀스  

### 🔹 동작 예시  
- **입력(Inputs)**: "나는 학생이다"  
- 인코더 스택(6개 모듈) → 문맥 벡터 생성  
- 디코더 스택(6개 모듈) → 문맥 벡터와 이전 단어 정보를 기반으로 최종 출력 생성  
- **출력(Output)**: "I am a student"  

### 🔹 특징  
- 여러 층을 거치며 점점 더 복잡하고 추상적인 문맥 정보 학습  
- 깊은 구조 덕분에 복잡한 문장 관계도 잘 표현 가능

---

<img width="817" height="477" alt="image" src="https://github.com/user-attachments/assets/80d0581a-250e-4355-b9eb-cea976307d34" />


### 🔹 인코더(Encoder) 내부 구조  
- 인코더는 두 가지 주요 서브레이어로 구성됨  
  1. **Multi-Head Attention**: 입력 토큰 간의 관계를 여러 시각(Head)에서 동시에 파악  
  2. **Feed Forward Network (FFN)**: Attention 결과를 비선형 변환으로 특징 강화  
- 각 서브레이어는 **Residual Connection**(잔차 연결)과 **Layer Normalization**을 거쳐  
  학습 안정성과 정보 손실 방지를 지원  

### 🔹 동작 흐름  
1. 입력 토큰 임베딩이 **Multi-Head Attention**으로 전달  
2. Attention 결과와 입력을 **Residual Connection**으로 더한 뒤 Layer Normalization 적용  
3. 결과를 **Feed Forward Network**에서 비선형 변환  
4. 변환 결과와 이전 출력을 다시 Residual Connection으로 더한 뒤 Layer Normalization 적용  
5. 다음 인코더 레이어로 전달  

---

<img width="840" height="544" alt="image" src="https://github.com/user-attachments/assets/c5e8e49d-2f31-4b0d-8081-c6e48bff1256" />


### 🔹 디코더(Decoder) 내부 구조  
- 디코더는 다음 세 가지 주요 서브레이어로 구성됨  
  1. **Masked Multi-Head Attention**: 미래 시점 단어를 보지 못하도록 마스킹하여 현재까지의 정보만 반영  
  2. **Multi-Head Attention**: 인코더 출력과 디코더의 현재 상태를 결합해 중요한 문맥 정보 반영  
  3. **Feed Forward Network (FFN)**: Attention 결과를 비선형 변환으로 특징 강화  
- 각 서브레이어는 **Residual Connection**과 **Layer Normalization**을 거쳐  
  안정적인 학습과 정보 보존을 지원  

### 🔹 동작 흐름  
1. 이전에 생성된 단어 시퀀스를 **Masked Multi-Head Attention**에 입력하고  
   마스킹을 통해 미래 단어 정보 차단  
2. 결과를 **Residual Connection + Layer Normalization** 처리  
3. 인코더 출력과 결합하여 **Multi-Head Attention** 수행  
4. 결과를 다시 Residual Connection + Layer Normalization 처리  
5. **Feed Forward Network**에서 비선형 변환 후 Residual Connection + Layer Normalization 적용  
6. 최종적으로 다음 시점 단어를 예측하여 출력  

---

<img width="1173" height="596" alt="image" src="https://github.com/user-attachments/assets/31214d8f-445d-4100-88bb-6ed1f023a482" />

---
<img width="1188" height="605" alt="image" src="https://github.com/user-attachments/assets/f5d8bfdb-fd6f-4f65-b4b6-81502ecc0ea4" />



### 🔹 Transformer 전체 Encoder-Decoder 구조  
1. **Embedding**  
   - 입력 토큰을 고정 길이의 밀집 벡터로 변환  
   - 학습 가능한 임베딩 테이블을 사용하여 각 단어를 벡터로 매핑  

2. **Positional Encoding**  
   - Transformer는 순차 처리 구조가 없기 때문에, 토큰의 위치 정보를 별도로 제공  
   - 사인/코사인 함수를 이용해 위치 정보를 임베딩 벡터에 더함  

3. **Encoder**  
   - 구성: **Self-Attention → Feed Forward**  
   - Self-Attention: 입력 시퀀스 내 토큰 간 관계 계산  
   - Feed Forward: Attention 결과를 비선형 변환하여 특징 강화  
   - 각 서브레이어 후 Residual Connection + Layer Normalization 적용  

4. **Decoder**  
   - 구성: **Masked Self-Attention → Encoder-Decoder Attention → Feed Forward**  
   - Masked Self-Attention: 미래 단어를 보지 못하게 마스킹  
   - Encoder-Decoder Attention: 인코더 출력과 디코더 상태를 결합해 문맥 반영  
   - Feed Forward: Attention 결과를 비선형 변환  
   - 각 서브레이어 후 Residual Connection + Layer Normalization 적용  

5. **Prediction**  
   - 디코더의 최종 출력을 Linear 변환 후 Softmax를 통해 각 단어의 등장 확률 계산  
   - 확률이 가장 높은 단어를 다음 시점 출력으로 선택
  
---

<img width="1342" height="420" alt="image" src="https://github.com/user-attachments/assets/7d14f94c-ba60-48ba-b35c-5c774e2ef9b5" />

<img width="1327" height="420" alt="image" src="https://github.com/user-attachments/assets/c5194bdb-d276-4432-8e79-fd248fa26e07" />

<img width="1330" height="420" alt="image" src="https://github.com/user-attachments/assets/db6b5038-4eb7-492f-a08b-28062a112be4" />


### 🔹 단어(Token) 형태 데이터를 수치로 변환 (Embedding)  
- 초기 입력은 **One-Hot Vector** 형태로 표현되며, 단어 사전에 있는 단어 개수(# of vocab)만큼 차원을 가짐  
- **Embedding Layer**를 통해 학습이 진행되며, 유사한 의미를 가진 단어들은 **유사한 벡터 값**을 갖도록 변환  
- 변환된 임베딩 벡터는 모델이 문맥을 이해하고 단어 간 관계를 파악하는 데 활용됨  

### 🔹 예시  
- 입력 문장: "나는 학생이다"  
  - 토큰별 One-Hot Vector 예:  
    - `나는` → [0, 1, 0, 0, 0, ...]  
    - `학생` → [0, 0, 0, 0, 1, 0, ...]  
    - `이다` → [0, 0, 1, 0, 0, ...]  
  - 각 토큰은 Embedding Layer를 거쳐 **차원 축소된 밀집 벡터**로 변환됨  
    - 예: `나는` → [0.12, -0.08, 0.53, ...]  
- 시각화 예: `cat`과 `kitten`은 가까운 벡터 위치, `dog`와도 근접 → 의미상 관련성 반영  

---

<img width="1213" height="429" alt="image" src="https://github.com/user-attachments/assets/2e104cf7-a2eb-462d-9400-dd30b701aa32" />

<img width="1300" height="335" alt="image" src="https://github.com/user-attachments/assets/b58895f1-e2e4-4dc8-9465-adaa956285d7" />


### 🔹 단어 간 순차성을 반영하기 위한 기법: Positional Encoding  
- Transformer는 RNN 계열과 달리 **입력값을 순차적으로 처리하지 않고 병렬 처리**를 수행  
- 이 때문에 입력 토큰 간의 **순서 정보가 자연스럽게 포함되지 않음**  
- 단어 간 순서를 반영하기 위해 **Positional Encoding**을 추가하여, 각 단어 벡터에 위치 정보를 더함  

### 🔹 동작 원리  
1. 각 단어는 먼저 **Word Embedding**을 통해 의미 벡터로 변환  
2. 각 위치에 해당하는 **Positional Encoding 벡터**를 생성  
3. Word Embedding 벡터와 Positional Encoding 벡터를 더해,  
   **의미 정보 + 위치 정보**가 모두 포함된 최종 임베딩을 생성  

### 🔹 예시  
- 입력 문장: "나는 학생이다"  
  - `나는`: 의미 벡터 + 위치(1) 벡터 → 순서 정보가 반영된 임베딩  
  - `학생`: 의미 벡터 + 위치(2) 벡터  
  - `이다`: 의미 벡터 + 위치(3) 벡터  

---

<img width="1403" height="475" alt="image" src="https://github.com/user-attachments/assets/f52b7ca7-672d-4d92-ae04-5afe98604707" />

<img width="1370" height="423" alt="image" src="https://github.com/user-attachments/assets/434659c7-90fb-4454-a3d4-04803e287922" />


### 🔹 주기함수를 활용한 Positional Encoding 구성  
- **Sinusoid Positional Encoding**: 사인(sin)과 코사인(cos) 함수를 이용해 위치 정보를 표현  
- 순차성을 부여하되 의미 정보(Word Embedding)가 변질되지 않도록 값의 범위를 **-1 ~ 1**로 유지  
- 모든 단어는 동일한 차원의 **Positional Encoding 벡터**를 가짐  
- 먼 단어 사이에는 큰 값, 가까운 단어 사이에는 작은 값이 나오도록 설계  

### 🔹 계산 방식  
$PE_{(pos, 2k)} = \sin\left( \frac{pos}{10000^{2k/d_{model}}} \right)$  

$PE_{(pos, 2k+1)} = \cos\left( \frac{pos}{10000^{2k/d_{model}}} \right)$  

- $pos$: 입력 시퀀스에서의 위치 (예: "나는"=0, "학생"=1, "이다"=2)  
- $i$: 임베딩 벡터 내 차원의 인덱스 (0, 1, 2, …)  
- $d_{model}$: 임베딩 벡터의 차원 수  

### 🔹 예시  
- 입력 문장: "나는 학생이다"  
- Word Embedding 벡터와 Positional Encoding 벡터를 더하여 **의미 + 위치 정보**가 포함된 최종 임베딩 생성  
- 예:  
  - `나는`: Word Embedding + Positional Encoding(위치 0)  
  - `학생`: Word Embedding + Positional Encoding(위치 1)  
  - `이다`: Word Embedding + Positional Encoding(위치 2)  

---

<img width="663" height="593" alt="image" src="https://github.com/user-attachments/assets/0dd3b168-5ea4-4e60-b812-ecdc0031dbb5" />


### 🔹 Positional Encoding의 특성  
- 가까운 위치의 단어끼리는 **유사도(1 - 거리)** 가 커서 값이 높게 나타남  
- 먼 위치의 단어 사이에는 유사도가 작아 값이 낮게 나타남  
- 이는 **순서 정보가 공간적으로 인접한 단어에 더 강하게 반영**되도록 하는 효과를 가짐  

### 🔹 시각화 예시  
- 가로축과 세로축은 **단어의 위치(Position)** 를 의미  
- 색이 진할수록 해당 위치 쌍의 **유사도 값이 큼**을 나타냄  
- 대각선 방향으로 진한 색이 나타나는 이유:  
  - 자기 자신과의 유사도는 최대  
  - 인접한 위치일수록 유사도가 높음 

---
# 이번주 강의 정리 및 다음 수업 예정
---

### 🔹 RNN 기반 방법론  
- 입력값을 **순차적으로 전달**받아 시계열 정보 반영 가능  
- 순차 처리 특성상 **병렬 처리 어려움** → 연산 효율 낮음  

### 🔹 Transformer의 입력 처리 방식  
- 입력을 순차적으로 받지 않고 **한 번에 처리** 가능 → 병렬 처리로 연산 효율 향상  
- **Positional Encoding**으로 시계열 정보 반영  
- **Self-Attention** 구조로 중요한 시점(단어) 정보 반영 가능  

### 🔹 Transformer의 활용 분야  
- **NLP**: 기계 번역, 텍스트 요약, 질의응답  
- **이미지 처리**: 이미지 인식, 객체 탐지  
- **제조 분야**: 상태 분류, 이상 탐지  

### 🔹 다음 수업 진도
-  Transformer의 Self-Attention 연산 과정 중 Query, Key, Value 벡터를 이용해 Attention Score를 계산하는 방법
