# 1️⃣ 다층 퍼셉트론

### 🔹 MLP의 기본 구조

다층 퍼셉트론(Multi-Layer Perceptron)은 다음과 같은 구조를 갖는다.

- **입력층(Input Layer)** : 외부에서 데이터를 받아들이는 층
- **은닉층(Hidden Layer)** : 비선형성을 학습하는 중간 층 (1개 이상 가능)
- **출력층(Output Layer)** : 최종 예측 결과를 내는 층

### 🔹 왜 은닉층(Hidden Layer)이 필요한가?

단일 퍼셉트론은 직선 하나로 데이터를 구분하는 **선형 분류기**이므로, XOR 같은 **비선형 문제**를 해결할 수 없다. 이 한계를 극복하기 위해, 입력과 출력 사이에 **은닉층(Hidden Layer)** 을 추가한다. 은닉층은 입력을 변환하여 **더 복잡한 함수 형태**를 모델링할 수 있도록 도와준다.

### 🔹 1960~1980년대 초 : 인공신경망 연구의 초기 단계

<img width="1020" height="501" alt="image" src="https://github.com/user-attachments/assets/aa71beed-22de-4539-9b12-7b338bd9d8df" />

- 이 시기에는 **단일 퍼셉트론(Perceptron)** 과 같은 **얕은 신경망(shallow network)** 구조에 연구의 초점이 맞춰져 있었다.
- 퍼셉트론은 단순한 선형 분류 문제는 해결할 수 있었지만, **비선형 문제(ex: XOR 문제)** 를 해결하지 못하는 한계가 있었다.

### 🔹 다층 퍼셉트론의 한계: 학습이 어려웠던 이유

<img width="1213" height="579" alt="image" src="https://github.com/user-attachments/assets/f5ef128d-5aae-4dfa-a264-1f43ef43a325" />

- XOR 문제와 같은 비선형 문제를 해결하기 위해 **은닉층(hidden layer)** 을 도입한 **다층 퍼셉트론(MLP)** 구조가 제안되었다.
- 하지만 이 당시에는 **역전파 알고리즘(backpropagation)** 이 개발되지 않아, 다음과 같은 문제가 존재했다:

  - 출력층의 오차는 계산할 수 있었지만, **어떤 은닉층 노드가 학습에 얼마나 기여했는지 알 수 없었다.**
  - 즉, **오차를 은닉층까지 전파하는 방법이 없었기 때문에**, 가중치 업데이트가 제대로 이루어지지 않았다.
  - 결과적으로 **다층 구조임에도 학습이 되지 않는** 구조적 문제가 있었다.

### 🔹 1986년: 역전파 알고리즘의 등장과 의미

- 1986년, Rumelhart, Hinton 등이 제안한 **역전파 알고리즘**은 이 문제를 해결했다.
- 역전파는 **출력층에서 계산된 오차를 연쇄적으로 은닉층으로 전달**하면서,  
  각 노드와 가중치에 대한 **기울기(gradient)** 를 계산할 수 있도록 한다.

- 즉, **"출력층만이 아니라 은닉층의 가중치도 정확하게 학습할 수 있게 해준 방법"** 이다.
- 이를 통해 **다층 퍼셉트론이 비로소 실질적인 학습이 가능해졌으며**,  
  이후 **심층 신경망(Deep Neural Network)** 으로 확장되는 기반이 마련되었다.

---
# 2️⃣ 역전파 알고리즘

<img width="726" height="362" alt="image" src="https://github.com/user-attachments/assets/3962ca53-ee1c-49a5-9d94-80051e0b2eb9" />


- 입력 $x_1$ = 0.5, $x_2$ = 0.3 이라 가정 
- 간단한 다층 신경망 2,2,1 가정
- 활성화 함수는 시그모이드 함수로 가정
- 손실함수는 MSE로 가정 ( $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$ )
- 모든 가중치를 랜덤하게 배치, lr = 0.1로 가정

### 🔹 역전파를 이용한 신경망 학습 (3단계)
- 1단계 : feedforward 순전파
- 2단계 : 손실 계산
- 3단계 : backpropagation 역전파

1단계부터 3단계를 반복하여 최적의 파라미터를 찾아가는 과정을 학습 (training)이라 부르기도 하고 최적화 (Optimization) 과정이라고 부르기도 함

---

### 🔹 1단계 : feedforward 순전파

<img width="613" height="376" alt="image" src="https://github.com/user-attachments/assets/9f29aaa8-4a80-4970-bc3a-bd36ea3a4b16" />

- 입력값에 대한 연결 가중치와 곱을 합하여 은닉층 노드에 넣는다.

<img width="571" height="452" alt="image" src="https://github.com/user-attachments/assets/ff32f7b9-1edf-4cd9-84d6-9a06d4d2b523" />

- 위 단계에서 넣었던 값을 은닉층 노드의 활성화 함수에 넣는다.

$$
z_1 = x_1 w_1 + x_2 w_3 = 0.5 \times 0.7 + 0.3 \times 0.4 = 0.47
$$

$$
h_1 = \sigma(z_1) = 0.615
$$

$$
z_2 = x_1 w_2 + x_2 w_4 = 0.5 \times 0.3 + 0.3 \times 0.6 = 0.33
$$

$$
h_2 = \sigma(z_2) = 0.582
$$

<img width="578" height="330" alt="image" src="https://github.com/user-attachments/assets/17440895-3275-4f83-a203-cafa6e90ce43" />

- 그 다음 연결 가중치와의 곱을 합하여 출력층 뉴런에 넣는다.

$$
h_1 = \sigma(z_1) = 0.615
$$

$$
h_2 = \sigma(z_2) = 0.582
$$

$$
z_3 = h_1 w_5 + h_2 w_6 = 0.615 \times 0.55 + 0.582 \times 0.45 = 0.6
$$

<img width="578" height="333" alt="image" src="https://github.com/user-attachments/assets/2a040d08-e52b-41d6-b32f-e7d8642fbf7b" />

- 마지막으로 출력층의 시그모이드 함수로 최종 출력값을 구한다.

$$
z_3 = h_1 w_5 + h_2 w_6 = 0.615 \times 0.55 + 0.582 \times 0.45 = 0.6
$$

$$
o_1 = \sigma(z_3) = 0.645
$$


### 🔹 2단계 : 손실 계산

<img width="1010" height="345" alt="image" src="https://github.com/user-attachments/assets/3e57fa61-45b3-4abd-adea-b505bef84422" />

- MSE를 사용하기로 가정했기 때문에 이 공식을 이용 -> $\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$
- 출력층 뉴런이 1개이기 때문에 n=1 -> $\text{MSE} = \frac{1}{1}  (y_1 - \hat{y}_1)^2$

- $\hat{y}$ 값은 0.645이며, y값(실제값)을 1이라고 가정하면 오차의 값은

$$
\text{MSE} = \frac{1}{1}  (1 - 0.645)^2 = 0.126
$$

- 따라서 오차 C = 0.126이다.

### 🔹 3단계 : backpropagation 역전파

<img width="584" height="370" alt="image" src="https://github.com/user-attachments/assets/64b68adf-ad1d-4199-a092-2857d1fd4529" />

- 우선 역전파를 이용해 가중치 w5를 업데이트 해보자.

- 경사하강법의 가중치 업데이트 공식은 다음과 같다. 여기서 C는 Cost function(손실함수)를 의미한다.

$$
\text{새 연결강도} = \text{현 연결강도} + {\color{red}(-\nabla C)} \times \text{학습률}
$$


- w5의 경우에는 그림과 같이 편미분 값으로 바꿀 수 있다.

$$
\text{새 연결강도} = \text{현 연결강도} + {\color{red}\frac{\partial C}{\partial w_5}} \times \text{학습률}
$$

- 여기서 $\frac{\partial C}{\partial w_5}$를 바로 구할 수 없기 때문에 아는 값들을 이용하여 계산을 위해 편미분(Chain Rule)을 사용한다.

---
- Chain Rule?

<img width="692" height="226" alt="image" src="https://github.com/user-attachments/assets/7cfdddb4-3fe6-4f0a-b279-f43a4ac6d954" />

  
- 우리가 어떤 두 변수의 미분값을 구하려 하나 그 관계를 모를 때, 각각 아는 미분 값들로 연쇄적으로 확장시켜 나가면 어려운 문제도 부분들을 해결하여 전체를 해결할 수 있다. 각각의 변수들이 사라지면 결국 구하고자 하는 관계만 남는 원리를 이용한다.

- 예를 들어 이런 문제가 있다고 하자.
	-  치타는 사자보다 2배 빠르고, 
	   사자는 곰보다 2배 빠르고, 
     곰은 사람보다 1.5배 빠르다고 할 때, 
     치타는 사람보다 몇 배 빠른가?

$$
\frac{d 치타}{d 사람} = \frac{d 치타}{d 사자} \cdot \frac{d 사자}{d 곰} \cdot \frac{d 곰}{d 사람}
$$

-  직관적으로 2 x 2 x 1.5 = 6배 빠름을 알 수 있다. 이것이 연쇄 미분의 개념이다.

---

- 다시 돌아와서 가중치 w5와 손실간의 변화량을 부분들로 나누면 다음과 같다.

$$
\frac{\partial C}{\partial w_5} = \frac{\partial C}{\partial o_1} \cdot \frac{\partial o_1}{\partial z_3} \cdot \frac{\partial z_3}{\partial w_5}
$$

- $\frac{\partial C}{\partial o_1}$ 계산

$$
C = (y - o_1)^2
$$

$$
\frac{\partial C}{\partial o_1} = -2(y - o_1)
$$

$$
\frac{\partial C}{\partial o_1} = -2(1 - 0.645) = -0.71
$$

- $\frac{\partial o_1}{\partial z_3}$ 계산 : sigmoid의 미분 값과 같으므로

$$
\frac{\partial o_1}{\partial z_3} = o(z)(1 - o(z)) = o_1(1-o_1) = 0.645(1-0.645) = 0.229
$$

- $\frac{\partial z_3}{\partial w_5}$ 계산

$$
z_3 = h_1 w_5 + h_2 w_6
$$

$$
\frac{\partial z_3}{\partial w_5} = \frac{\partial}{\partial w_5}(h_1 w_5 + h_2 w_6) = h_1 = 0.615
$$

- $\frac{\partial c}{\partial w_5} = -0.71 \cdot 0.229 \cdot 0.615 = \color{red}{-0.01}$

- 경사강법의 가중치 업데이트 공식에 따라서 

$$
\text{새 연결강도} = \text{현 연결강도} + \frac{\partial C}{\partial w_5} \times \text{학습률}
$$

$$
\text{새 연결강도} = 0.55 + \text{-0.01} \times \text{0.1} = 0.551
$$

이를 통해 w5를 업데이트 할 수 있고, 나머지 가중치 또한 이와 같은 방식으로 업데이트 가능하다.




# 3️⃣ Pytorch

### 🔹 Pytorch란?

PyTorch는 Facebook에서 개발한 **딥러닝 프레임워크**로, 유연하고 사용하기 쉬운 인터페이스를 제공한다.  
Python과의 호환성이 뛰어나고, 연구 및 실험 단계에서 널리 활용되며, 학습, 모델링, 추론 등을 위한 다양한 기능을 포함한다.

공식 사이트: [https://pytorch.org](https://pytorch.org)

### 🔹 주요 특징

- **동적 계산 그래프 (Dynamic Computational Graph)**  
  런타임에 계산 그래프가 정의되어, 디버깅과 실험에 매우 유리하다.

- **NumPy와 유사한 텐서(Tensor) 연산 지원**  
  GPU 가속이 가능한 텐서 연산 구조 제공.

- **autograd를 통한 자동 미분 지원**  
  역전파 계산을 자동으로 처리함.

- **torch.nn 모듈을 통한 신경망 구성 지원**  
  다양한 레이어, 손실함수, 최적화 알고리즘 제공.

- **GPU 지원 (CUDA)**  
  `.to("cuda")` 또는 `.cuda()`를 통해 간단하게 GPU 연산 수행 가능.

  
### 🔹 기본 예제

```python
import torch

# 텐서 정의
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# 선형 연산
y = x @ w + b  # 또는 torch.dot(x, w) + b

# 손실 함수 예: 제곱 오차
loss = (y - 1.0) ** 2

# 역전파
loss.backward()

# 기울기 확인
print(w.grad)
print(b.grad)
```

- Pytorch 프레임워크를 사용하기 위해서는 `import torch` 를 해주고 사용 가능하다.

- 텐서 정의
	- `requires_grad=True`는 해당 텐서가 **미분 대상**임을 의미함.
	- 학습 대상인 입력 `x`, 가중치 `w`, 편향 `b`를 선언.
 
- 선형 연산
	- `@` 연산자는 벡터 내적(dot product)을 의미.
	- 결과 `y`는 스칼라 값.

- 손실 함수
	- 원하는 정답 `1.0`과의 **제곱 오차(MSE)** 를 사용.
 
- 역전파 수행
	- `loss`를 기준으로 모든 파라미터에 대해 **기울기(gradient)** 를 자동으로 계산.
	- 계산된 결과는 `w.grad`, `b.grad`에 저장됨.

- 기울기 출력
	- 각 파라미터에 대해 손실 함수의 **기울기(gradient)** 가 출력됨.

이와 같이 Pytorch의 `.backward()` 한 줄로 자동 미분을 수행할 수 있으며,  
자세한 내용은 [https://pytorch.org](https://pytorch.org) 에서 공식 문서를 참고하면 된다.

---

### 🔹 수동 미분과 자동 미분 비교의 필요성
딥러닝에서 가장 핵심적인 학습 과정은 역전파(backpropagation) 를 통한 기울기 계산이다. 이 기울기는 모델의 파라미터를 어떻게 조정해야 손실(loss)이 줄어들지를 알려주는 중요한 정보이다.

아래 실습에서는 넘파이(Numpy) 를 사용해 직접 수동으로 미분식을 구현한 방식과,
파이토치(PyTorch) 의 autograd 기능을 이용한 자동 미분(automatic differentiation) 방식을 비교하여,
두 방식이 실제로 유사한 결과를 낸다는 점을 실습을 통해 확인해보고자 한다.


### 🔹 넘파이 vs 파이토치 2단 MLP 비교

#### 1️. 넘파이로 (수동 미분)

---
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

---

#### 2. 수치 미분으로 검증

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

#### 3. 파이토치로 (자동 미분)

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

# 4️⃣ 과제

### 🔹 과제 1) PyTorch에서 제공하는 Optimizer 종류 조사

아래 내용을 위주로 조사해보시면 됩니다.

- 대표적인 옵티마이저 SGD, Adam, RMSprop, Adagrad 등의 알고리즘 소개
- 각 옵티마이저의 동작 방식 요약 (수식 포함 가능)
- 각 옵티마이저의 장단점 2가지 이상 서술
- 어떤 상황에서 어떤 옵티마이저를 선택하는 것이 좋은지 정리
