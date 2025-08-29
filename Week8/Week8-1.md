---
# 저번 수업 복습
---

<img width="1305" height="554" alt="image" src="https://github.com/user-attachments/assets/30504078-c78c-4fb3-9d6e-cf47e16610a3" />



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

<img width="864" height="549" alt="image" src="https://github.com/user-attachments/assets/d98ee981-6d1b-4d24-b8b5-ba5065aac50f" />



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

<img width="845" height="531" alt="image" src="https://github.com/user-attachments/assets/9667c7b6-8243-4a41-85e0-4581a5518f54" />



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

<img width="817" height="477" alt="image" src="https://github.com/user-attachments/assets/873e9095-07e0-4d53-94e3-d6a8fed67857" />


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

<img width="840" height="544" alt="image" src="https://github.com/user-attachments/assets/7e9e99d8-e65d-44df-8538-7436e21523a1" />


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

<img width="1173" height="596" alt="image" src="https://github.com/user-attachments/assets/2433a345-68a5-4b2e-b35e-637f3f3f3f72" />

<img width="1188" height="605" alt="image" src="https://github.com/user-attachments/assets/9c0fcc62-ceee-4004-aa3b-fabbb05c8831" />



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

<img width="1342" height="420" alt="image" src="https://github.com/user-attachments/assets/b4939995-2044-45ad-9ac6-f5899e359a5d" />

<img width="1327" height="420" alt="image" src="https://github.com/user-attachments/assets/41271777-28cc-4a6e-a8f6-027ee02d3b11" />

<img width="1330" height="420" alt="image" src="https://github.com/user-attachments/assets/33ec5410-ba19-4da6-afa7-fe1ada090808" />


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

<img width="1213" height="429" alt="image" src="https://github.com/user-attachments/assets/ef7f1749-f0d9-4454-946d-617498cb9d5a" />

<img width="1300" height="335" alt="image" src="https://github.com/user-attachments/assets/c572463f-e4c2-4f1c-856f-4a11a48c64c5" />


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

<img width="1403" height="475" alt="image" src="https://github.com/user-attachments/assets/f0feb830-c0da-44e5-8f3c-2625d47c8912" />

<img width="1370" height="423" alt="image" src="https://github.com/user-attachments/assets/222ef91b-2c74-4b80-ba9d-053ff2bfd65a" />


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

<img width="663" height="593" alt="image" src="https://github.com/user-attachments/assets/7312e174-e2c8-4543-a2e1-a4663e32de21" />


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
# 이번 주차 수업
---

<img width="1194" height="609" alt="image" src="https://github.com/user-attachments/assets/611be082-64b4-40a5-a259-8e39af0ffc4e" />


### 🔹 Encoder 핵심 구성 요소  
- **Self-Attention**: 입력 시퀀스 내 토큰 간 관계를 계산하여 문맥 정보 생성  
- **Feed Forward**: Attention 결과를 비선형 변환으로 특징 강화  
- **Add & Norm**: 출력에 입력을 더한 뒤 정규화하여 학습 안정성 확보  

### 🔹 Encoder의 역할  
- 입력 문장을 **문맥(Context) 벡터**로 변환  
- 생성된 문맥 벡터는 디코더의 **Encoder-Decoder Attention** 단계로 전달되어 최종 출력 생성에 활용  

<img width="1343" height="310" alt="image" src="https://github.com/user-attachments/assets/8cca82f9-8f1d-45be-a3a8-78bdb70339c3" />



### 🔹 Encoder: Self-Attention 개요  
- 문장 내 **모든 단어의 관계를 비교**하여 중요한 특징을 추출하고, 이를 통해 출력 벡터 $z$를 생성  
- 각 단어는 **Query(Q)**, **Key(K)**, **Value(V)** 세 가지 형태로 변환되며, 변환을 위한 가중치 행렬($W^Q_E$, $W^K_E$, $W^V_E$)은 학습을 통해 결정  
- 각 행렬의 크기는 **하이퍼파라미터**로 설정되어 모델 구조를 결정  

---

### 🔹 Q, K, V의 의미  
- **Query (Q)**: 현재 토큰이 “어떤 정보를 찾고 싶은지”를 나타내는 벡터 (질문 역할)  
- **Key (K)**: 각 토큰이 “어떤 정보를 가지고 있는지”를 나타내는 벡터 (비교 대상 역할)  
- **Value (V)**: 실제 전달되는 정보 벡터 (최종 결과 생성에 사용)  

### 🔹 왜 3개로 나누는가  
- Q와 K는 **유사도(Attention Score)** 계산에 사용  
- V는 계산된 가중치를 반영해 최종 출력 생성  
- 역할을 분리함으로써, 같은 단어라도 상황에 따라 다른 방식으로 정보 추출 가능  
- 구조적으로 “**어떤 기준(Q)**으로, **어떤 정보(K)**를 비교하고, 그 결과로 **무엇(V)**을 가져올지”를 명확히 구분  

---

<img width="962" height="603" alt="image" src="https://github.com/user-attachments/assets/6ac1f945-fec2-4e4c-abba-285a2a71d870" />


### 🔹 직관적 예시 ("나는 학생이다")  
1. **Q**: `나는`이 다른 단어(`학생`, `이다`)와 어떤 관계가 있는지 파악  
2. **K**: 각 단어(`나는`, `학생`, `이다`)가 가진 문맥 속 특징  
3. **V**: 각 단어가 가진 실제 의미 벡터  
4. Q와 K로 단어 간 연관도를 계산 → 해당 가중치를 V에 곱해 최종 문맥 벡터 $Z$ 생성  

---

<img width="1385" height="569" alt="image" src="https://github.com/user-attachments/assets/511a1295-678a-401f-a2b7-bf38d7e4ac6a" />


<img width="1385" height="576" alt="image" src="https://github.com/user-attachments/assets/f0d0e434-2d8e-4742-8071-26ff1e24bf11" />


### 🔹 Q, K, V 벡터 생성 과정  
- 입력 시퀀스 X는 각 단어의 **임베딩 + 위치정보**를 포함한 행렬 (예: 3×4)  
- Self-Attention에서는 X를 세 가지 다른 가중치 행렬과 곱해 **Q, K, V**를 생성  
  - $Q = X \times W^Q_E$  → (3×2)  
  - $K = X \times W^K_E$  → (3×2)  
  - $V = X \times W^V_E$  → (3×2)  
- 각 가중치 행렬($W^Q_E$, $W^K_E$, $W^V_E$)은 학습을 통해 최적화되며, **서로 다른 역할**을 수행  
  - **Q**: 현재 단어가 어떤 정보를 찾고 싶은지 표현  
  - **K**: 각 단어가 가진 정보의 속성 표현  
  - **V**: 최종적으로 전달할 정보 표현  
- 모든 연산은 **행렬 단위**로 수행되어 연산 효율성과 병렬 처리가 가능

---

<img width="1327" height="598" alt="image" src="https://github.com/user-attachments/assets/f7a1ced6-f9b8-42ff-91d9-7b29c793d0c5" />


### 🔹 Multi-Head Attention 출력 결합 과정  
- **Multi-Head Attention**에서는 여러 개의 head(예: 8개)를 사용하여 서로 다른 시각에서 Attention을 수행  
- 각 head에서 도출된 출력 $Z_1, Z_2, \dots, Z_8$을 **행렬 결합(Concatenate)** 하여 하나의 큰 행렬 생성  
  - 예: 8개 head × 각 head 출력 크기(3×2) → 결합 후 (3×16)  
- 결합된 행렬에 가중치 행렬 $W_0$ (크기 16×4)을 곱해, **입력과 동일한 차원(3×4)** 의 최종 출력 $Z$ 생성  

### 🔹 특징  
- 여러 head의 출력을 결합함으로써 **다양한 관점의 문맥 정보**를 하나의 벡터에 반영  
- $W_0$를 적용하여 차원 축소 및 입력 차원과의 일관성 유지  
- Transformer는 **공간적(spatial)**, **시간적(temporal)** 구조를 유지하면서 정보 처리

---

<img width="1410" height="481" alt="image" src="https://github.com/user-attachments/assets/0e9cf2e2-6b1f-4fdb-8da8-9804481d0ee6" />

### 🔹 Self-Attention 후처리 과정  
- **Encoded Vector ($z$)**: Self-Attention 계산 결과, 각 단어별로 문맥이 반영된 벡터 생성  
- **Residual Connection**: 입력 벡터($x$)와 Attention 결과($z$)를 더하여 정보 손실 방지 및 학습 안정성 강화  
  - 예: $x_1 + z_1$, $x_2 + z_2$, $x_3 + z_3$  
- **Layer Normalization**: Residual Connection 결과를 정규화하여 학습 안정성 향상  
  - 최종 출력: $z^s_1$, $z^s_2$, $z^s_3$  

💡 $d_k = \frac{d_{model}}{\text{num heads}}$  
- 예: $d_{model} = 512$, head 수 = 8 → $d_k = 64$  
- 각 head가 다루는 차원을 균등 분할하여 계산 효율성과 다중 관점 학습 가능

---

<img width="1420" height="532" alt="image" src="https://github.com/user-attachments/assets/0501cd08-d3bf-46ac-9768-9bff03ca5706" />

### 🔹 Feed Forward Network (FFN)  
- Self-Attention 결과($z^s$)에 **비선형 변환**을 적용하여 각 단어의 표현을 강화  
- 동일한 Encoder 블록 내에서는 모든 FFN이 **같은 파라미터**를 사용하여 연산 효율성 확보  
- 다른 Encoder 블록 간에는 **서로 다른 파라미터**를 사용해 계층별 학습 다양성 제공  

### 🔹 처리 과정  
1. **입력**: Self-Attention과 Layer Norm을 거친 출력 $z^s$  
2. **Feed Forward 연산**: 두 개의 Fully Connected Layer와 활성화 함수를 사용하여 각 단어 벡터 개별 변환  
3. **Residual Connection**: FFN 출력($z^{s1}$)과 입력($z^s$)을 더해 정보 손실 방지  
4. **Layer Normalization**: Residual 결과를 정규화하여 학습 안정성과 수렴 속도 향상  
5. **출력**: 최종 $z^{s11}$ 벡터가 다음 Encoder 레이어 또는 Decoder로 전달됨


---

<img width="1206" height="622" alt="image" src="https://github.com/user-attachments/assets/31ee4ea2-e5d0-4e83-b98d-4f6c3acf7594" />

### 🔹 Decoder 구조 개요  
- **입력**: 이전 시점에서 생성된 출력 토큰 시퀀스 (오른쪽으로 한 칸 shift)  
- **Positional Encoding**을 적용하여 순서 정보 반영  
- 세 가지 핵심 서브 레이어로 구성:  
  1. **Masked Self-Attention**  
     - 현재 시점 이후의 단어를 보지 못하도록 마스킹 처리  
     - 디코더가 미래 단어 정보를 사용하지 않게 하여 **자동회귀(auto-regressive)** 특성 유지  
  2. **Encoder-Decoder Attention**  
     - 인코더에서 생성된 문맥 벡터와 디코더의 현재 상태를 결합  
     - 입력 문장과의 관계를 고려하여 더 정확한 출력 생성  
  3. **Feed Forward**  
     - Attention 결과를 비선형 변환하여 표현력 강화  

### 🔹 처리 흐름  
1. 이전 시점의 출력 시퀀스를 **Masked Self-Attention**에 입력  
2. Encoder 출력과 결합하여 **Encoder-Decoder Attention** 수행  
3. 결과를 **Feed Forward Network**로 변환  
4. 각 서브 레이어마다 **Residual Connection + Layer Normalization**을 거쳐 안정적 학습 지원  

### 🔹 출력  
- 디코더의 최종 출력은 Linear → Softmax로 변환되어 다음 단어의 확률 분포를 생성

---

<img width="1399" height="558" alt="image" src="https://github.com/user-attachments/assets/15206bd5-e158-467c-9b85-2444a2bec06b" />

### 🔹 Masked Self-Attention 개념  
- 디코더에서 현재 시점까지 주어진 단어들만 참조하여 Attention을 수행  
- **미래 시점 마스킹 이유**: 모델이 순차적으로 단어를 예측하도록 하여, 이전 예측 단어를 기반으로 다음 단어를 생성하게 함  
- 마스킹된 위치의 Score는 $-\infty$로 설정되어 Softmax 결과가 0이 되도록 처리

### 🔹 처리 과정 ("I am a student") 예시  
1. **입력 임베딩 + 위치정보**  
   - $x_1$ (`I`), $x_2$ (`am`), $x_3$ (`a`), $x_4$ (`student`)  
2. **Q, K, V 생성 (디코더 전용 가중치 사용)**  
   - $Q = X \times W^Q_D$  
   - $K = X \times W^K_D$  
   - $V = X \times W^V_D$  
   - $d_k = \frac{512}{8} = 64$  
3. **유사도(Score) 계산 및 마스킹 적용**  
   - 현재 단어의 $Q$와 과거 시점 단어들의 $K$만 비교  
   - 미래 단어(`am` 이후)는 $-\infty$ 처리 → Softmax 시 확률 0  
   - 예: $q_1 \cdot k_1 = 152$, $q_1 \cdot k_2 = -\infty$, $q_1 \cdot k_3 = -\infty$, $q_1 \cdot k_4 = -\infty$  
4. **Softmax + 가중합**  
   - Softmax 결과를 해당 $V$ 값과 곱한 뒤 합산하여 $z_1$ 생성  
5. **출력**  
   - 최종 **Decoded Vector**는 입력과 동일한 차원을 유지하며, 다음 레이어로 전달  

<img width="1397" height="553" alt="image" src="https://github.com/user-attachments/assets/0453ab00-6fc9-4496-9540-f2c29722aa45" />

### 🔹 Masked Self-Attention (두 번째 시점: "am")
- **현재 시점($x_2$: "am")**에서의 Query($q_2$)는 **과거 단어**들의 Key($k_1$, $k_2$)와만 유사도 계산  
- **미래 단어**($x_3$: "a", $x_4$: "student")의 Key는 마스킹되어 $-\infty$ 처리 → Softmax 결과 0

### 🔹 계산 과정
1. **Q, K, V 생성**  
   - $q_2$, $k_1$, $k_2$, $v_1$, $v_2$는 디코더 전용 가중치로부터 생성  
2. **Score 계산**  
   - $q_2 \cdot k_1 = 102$  
   - $q_2 \cdot k_2 = 136$  
   - $q_2 \cdot k_3 = -\infty$, $q_2 \cdot k_4 = -\infty$  
3. **스케일 조정**  
   - $102 / \sqrt{64} = 102 / 8 = 13$  
   - $136 / 8 = 17$  
4. **Softmax 적용**  
   - $\text{Softmax}([13, 17, -\infty, -\infty]) = [0.018, 0.982, 0, 0]$  
5. **가중합 연산**  
   - $0.018 \times v'_1 + 0.982 \times v'_2 = z_2$  
6. **출력**  
   - $z_2^M$은 입력과 동일 차원을 갖는 **Decoded Vector**로 다음 레이어로 전달  

---

<img width="1406" height="506" alt="image" src="https://github.com/user-attachments/assets/ba1430d9-0fd3-4468-affc-a1517e7071ee" />

### 🔹 Encoder-Decoder Attention (두 번째 시점: "am")
- **Query($q_2$)**: 디코더의 Masked Self-Attention 결과 $z_2^M$에서 생성  
- **Key($k_1, k_2, k_3, k_4$)**, **Value($v_1, v_2, v_3, v_4$)**: **인코더 출력**에서 생성  
- **Self-Attention과 차이점**: Key와 Value가 **인코더**에서 오고, Query는 **디코더**에서 옴 → **cross-attention**  
### 🔹 계산 과정
1. **Q, K, V 준비**
   - $q_2$: 디코더에서 생성  
   - $k_1 \sim k_4$, $v_1 \sim v_4$: 인코더에서 생성
2. **Score 계산**
   - $q_2 \cdot k_1 = 98$  
   - $q_2 \cdot k_2 = 67$  
   - $q_2 \cdot k_3 = 102$  
   - $q_2 \cdot k_4 = -$ (생략)
3. **스케일 조정**
   - $98 / \sqrt{64} = 12$  
   - $67 / 8 = 8$  
   - $102 / 8 = 13$
4. **Softmax 적용**
   - $\text{Softmax}([12, 8, 13]) = [0.268, 0.005, 0.727]$
5. **가중합 연산**
   - $0.268 \times v'_1 + 0.005 \times v'_2 + 0.727 \times v'_3 = z_2$
6. **출력**
   - $z_2$는 다음 **Feed Forward Network**로 전달

---

<img width="1234" height="441" alt="image" src="https://github.com/user-attachments/assets/b7f72943-9e02-43b8-8425-c8261244e385" />

### 🔹 Decoder: Prediction
- **목적**: 디코더의 마지막 출력 벡터를 사용해 다음 토큰(단어)을 예측  
- **변형 가능성**: 수행하려는 task(번역, 요약, 질의응답 등)에 따라 Linear 레이어 이후 구조가 달라질 수 있음  

### 🔹 구성 요소
1. **Query** ($q_t$): 디코더의 Masked Self-Attention 결과 $z_t^M$
2. **Key / Value**: 인코더 출력에서 생성 ($k_i$, $v_i$)
3. **Encoder-Decoder Attention**을 통해 $z_t^{ED}$ 생성
4. **Summation**: Value의 가중합으로 Attention 출력 생성
5. **Linear Layer**:  
   - $y_t = z_t^{ED} W_l + b_l$  
   - 어휘 크기(vocabulary size)만큼 차원을 변환
6. **Softmax**:  
   - 각 단어에 대한 확률 분포 계산
   - 예: `[0, 0, 1, 0, 0, 0, ...]` → 가장 높은 확률의 단어 선택
7. **토큰 예측**:  
   - 예: "am" 다음에 "a"가 나올 확률이 가장 높으면 `"a"` 선택
  
### 🔹 예시 흐름 ("am" 다음 단어 예측)
1. Query: $q_2$ ("am"에 해당)  
2. Key, Value: 인코더에서 가져온 $k_1, k_2, k_3$, $v_1, v_2, v_3$  
3. Attention 연산 → $z_2^{ED}$  
4. Linear 변환 및 Softmax → 어휘 확률 분포 생성  
5. 최고 확률 토큰 = `"a"`

---


