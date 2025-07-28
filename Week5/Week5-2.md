# 저번 주차 복습

---

### 🔹 3차원 데이터의 합성곱 연산

- 이미지만 해도 세로, 가로에 더해서 채널까지 고려한 3차원 데이터이다. 2차원 일때와 비교하면, 길이 방향(채널 방향)으로 특징 맵(feature map)이 늘어난다.
- 채널 쪽으로 특징 맵이 여러 개 있다면 입력 데이터와 필터의 합성곱 연산을 채널마다 수행하고, 그 결과를 더해서 하나의 출력을 얻는다.
- 3차원의 합성곱 연산에서 주의할 점은 입력데이터의 채널수와 필터의 채널수가 같아야 한다.

<img width="685" height="937" alt="image" src="https://github.com/user-attachments/assets/8c5f7d21-54cd-4719-bc3e-8bd02b579e06" />

### 🔹 블록으로 생각하기

- 3차원 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각하면 된다.
- 입력데이터(C, H, W), 필터(C, FH, FW) = 출력데이터(1,OH,OW)
  - C : Channel 의미
  - FH : Filter Height
  - FW : filter Width

<img width="746" height="235" alt="image" src="https://github.com/user-attachments/assets/0d556009-d3a7-419d-917b-823572cf5473" />

---

- 그렇다면 합성 곱 연산의 출력으로 다수의 채널을 보내려면?
  - ->다수의 필터를 사용한다.

- 사용된 필터의 개수만큼 출력 데이터의 채널수 출력

<img width="767" height="448" alt="image" src="https://github.com/user-attachments/assets/d4beb84c-7e19-4113-b8ab-e5a206a1df2c" />

- 입력데이터 (C, H, W) * 필터 (C, FH, FW, FN) = 출력데이터(FN, OH, OW)
- 여기서 필터를 FN개 적용하면 출력 맵도 FN개 생성된다. 

- 위의 그림에서 보는 것과 같이 합성곱 연산에서는 필터의 수도 고려해야 한다. 
- 그런 이유로 필터의 가중치 데이터는 4차원 데이터이며, (출력 채널수, 입력 채널수, 높이, 너비) 순으로 순으로 쓰인다.

ex) 채널수 3, 크기 5x5 필터 20개, (20, 3, 5, 5)

 

합성곱 연산에도(완전 연결 계층과 마찬가지로) 편향이 쓰인다.
입력데이터 (C, H, W) * 필터 (FN, C, FH, FW) => 출력데이터(FN, OH, OW) + 편향 (FN, 1, 1) => (FN, OH, OW)
형상이 다른 블록의 덧셈은 넘파이의 브로드캐스트 기능으로 쉽게 구현이 가능하다.

<img width="798" height="319" alt="image" src="https://github.com/user-attachments/assets/bf62f708-4e10-49ad-9740-6bc4b5a5c8d5" />


### 🔹 배치 처리

- 신경망 처리에서는 입력 데이터를 한 덩어리로 묶어 배치를 처리했다.
- 완전 연결 신경망을 구현하면서는 이 방식을 지원하여 처리 효율을 높이고, 미니배치 방식의 학습도 지원하였다.
- 합성곱 연산도 마찬가지로 배치 처리를 지원하며, 각 계층을 흐르느 데이터의 차원을 하나 늘려 4차원 데이터로 저장한다.

<img width="912" height="310" alt="image" src="https://github.com/user-attachments/assets/509337ca-8cc0-464b-91de-268bab50bd6f" />

- 이처럼 데이터는 4차원 형상을 가진 채 각 계층을 타고 흐른다.
- 여기에서 주의할 점으로는 신경망에 4차원 데이터가 하나 흐를 때마다 데이터 N개 대한 합성곱 연산이 이뤄진다는 것이다.
  - 즉, N 회 분의 처리를 한 번에 수행하는 것이다

### 🔹 풀링 계층

- 풀링은 세로 가로 방향 공간을 줄이는 연산이다.
- 최대 풀링(Max Pooling)은 대상 영역에서 최댓값을 취하는 연산인 반면, 평균 풀링(Average Pooling)은 대상 연역에서 평균을 계산한다.
- 이미지 인식 분야에서는 주로 최대 풀링을 사용한다.

<img width="880" height="354" alt="image" src="https://github.com/user-attachments/assets/3c6b12af-8f36-440a-86a1-d058593acb93" />

- 풀링 계층의 특징(3)
  - 1. 학습해야 할 매개변수가 없다.
       - 풀링은 대상 영역에서 최댓값이나 평균을 취하는 명확한 처리이므로 특별히 학습할 것이 없다.
  - 2. 채널 수가 변하지 않는다.
       - 풀링 연산은 입력 데이터의 채널 수 그대로 출력데이터로 내보낸다.
       - 채널마다 독립적으로 계산하기 때문이다.
  - 3. 입력의 변화에 영향을 적게 받는다. (강건하다)
       - 입력데이터가 조금 변해도 풀링의 결과는 잘 변하지 않는다.
 
<img width="850" height="215" alt="image" src="https://github.com/user-attachments/assets/efcfcbbc-2994-4cf6-8c37-e5cae6d3d642" />



---
# 이번 주차 진도
---

### 🔹 4차원 배열

- 앞에서 설명한대로 CNN에서 계층 사이를 흐르는 데이터는 4차원이다.
   - 예) 데이터의 형상이 (10,1,28,28)인 경우
   - 높이 28, 너비 28, 채널 1개인 데이터가 10개
   - 이를 파이썬으로 구현하면 다음과 같다.
 
```python
import numpy as np

x=np.random.rand(10,1,28,28)
x.shape
```

  - 여기서 10개 중 첫번째 데이터에 접근하려면 단순히 x[0]이라고 쓴다.
  - 두 번째 데이터는 x[1] 위치에 있다.

```python
x[0].shape    # (1,28,28)
```

```python
x[1].shape    # (1,28,28)
```

```python
x[0.0].shape    # (28,28)
```


<img width="708" height="473" alt="image" src="https://github.com/user-attachments/assets/07947b3a-09c0-4e76-a6b6-fbe2eae3ab5d" />


- 첫 번째 데이터의 첫 채넣의 공간 데이터에 접근


```python
x[0,0]    # 또는 x[0][0]
```

- 이처럼 CNN은 4차원 데이터를 다룬다.

---

### 🔹 im2col로 데이터 전개

- 4차원 데이터를 다루는 CNN은 연산이 복잡해질 것 같지만 im2col 함수를 통해 간단하게 구현이 가능하다.
- im2col 함수를 없이 합성곱 연산을 구현한다면 for 문을 겹겹이 써야하는데, 이는 성능이 떨어지므로 추천되는 방식이 아니다.
- 또한 넘파이에서는 원소에 접근할 때 for 문을 사용하지 않는 편이 바람직하다.
- 따라서 im2col 함수를 사용하여 간단하게 표현하는데, im2col 함수란 무엇일까?

### 🔹 im2col 함수란?

- im2col은 입력 데이터를 필터링(가중치 계산) 하기 좋게 전개하는 함수이다.
- 아래 그림과 같이 3차원 입력 데이터에 im2col을 적용하면 2차원 행렬로 바뀐다.
- 정확히는 배치 안의 데이터 수까지 포함한 4차원 데이터를 2차원으로 변환한다.

<img width="773" height="343" alt="image" src="https://github.com/user-attachments/assets/b533b4a9-cdcc-43b6-8d73-4e4f1d5b0802" />

- im2col은 필터링 하기 좋게 입력 데이터를 전개합니다.
- 구체적으로는 아래 그림과 같이 입력데이터에서 필터를 적용하는 영역(3차원 블록)을 한 줄로 늘어 놓는다.
- 이 전개를 필터를 적용하는 모든 영역에서 수행하는 것이 im2col이다.

<img width="766" height="341" alt="image" src="https://github.com/user-attachments/assets/abd38370-bfc8-48df-ac30-5565b97eaba1" />

그림에서는 보기에 좋게끔 스트라이드를 크게 잡아 필터의 적용 영역이 겹치지 않도록 했지만, 실제 상황에서는 영역이 겹치는 경우가 대부분이다.
필터 적용 영역이 겹치게 되면 im2col로 전개한 후의 원소 수가 원래 블록의 원소 수보다 많아진다.
그래서 im2col을 사용해 구현하면 메모리를 더 많이 소비하는 단점이 있다.
하지만 컴퓨터는 큰 행렬을 묶어서 계산하는 데 탁월하다.

예를 들어 행렬 계산 라이브러리 등은 행렬 계산에 고도로 최적화되어 큰 행렬의 곱셈을 빠르게 계산할 수 있다.
im2col은 'image to column' 즉 이미지에서 행렬로라는 뜻으로, 딥러닝 프레임워크는 im2col이라는 이름의 함수를 만들어 합성곱 계층을 구현할 때 이용한다.
im2col로 입력 데이터를 전개한 다음에는 합성곱 계층의 필터(가중치)를 1열로 전개하고, 두 행렬의 곱을 계산하면 된다.
이는 완전연결 계층의 Affine 계층에서 한 것과 거의 같다.

<img width="762" height="441" alt="image" src="https://github.com/user-attachments/assets/5b3a1785-e79e-45bc-8548-1bc2cb5e3fa1" />

필터를 세로로 1열로 전개하고, im2col이 전개한 데이터와 행렬 곱을 계산하고, 마지막으로 출력 데이터를 변형한다.
위 그림과 같이 im2col 방식으로 출력한 결과는 2차원 행렬이다.
CNN은 데이터를 4차원 배열로 저장하므로 2차원인 출력 데이터를 4차원으로 변형한다.

### 🔹 합성곱 계층 구현하기

-  im2col 함수는 '필터 크기', '스트라이드'. '패딩'을 고려하여 입력 데이터를 2차원 배열로 전개하는 역할을 한다. 실제로 적용하면 다음과 같다.

`im2col(input_data, filter_h, filter_w, stride=1, pad=0)`

여기서
- input_data :  (데이터의 수, 채널 수, 높이, 너비)의 4차원 배열로 이뤄진 입력 데이터
- filter_h : 필터의 높이
- filter_w : 필터의 너비
- stride : 스트라이드
- pad : 패딩

이러면 im2col은 필터 크기와 스트라이드, 패딩을 고려하여 입력 데이터를 2차원 배열로 전개한다.
이제 실제로 사용한 예시는 다음과 같다.

---

```python
import numpy as np
  
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
  
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
  
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
  
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

x1 = np.random.rand(1, 3, 7, 7) # 데이터 수, 채널 수, 높이, 너비 
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape) # (9, 75) 
x2 = np.random.rand(10, 3, 7, 7) # 데이터 10개 
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90, 75)
```

여기에서는 두 가지 예를 보여주고 있다.
첫 번째는 배치 크기가 1(데이터 1개), 채널은 3개, 높이 너비가 7 X 7의 데이터이고,
두 번째는 배치 크기만 10이고 나머지는 첫 번째와 같다.

im2col 함수를 적용한 두 경우 모두 2번째 차원의 원소는 75개이다.
이 값은 필터의 원소 수와 동일한데(채널 3개, 5 X 5 데이터), 
배치 크기가 1일 때는 im2col의 결과의 크기가 (9, 75)이고, 
10일 때는 그 10배의 (90, 75) 크기의 데이터가 저장된다.

이제 이 im2col을 사용하여 합성곱 계층을 Convolution이라는 클래스로 구현해보자.

```python
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T  # 필터 전개
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        return out
```

 합성곱 계층은 필터(가중치), 편향, 스트라이드, 패딩을 인수로 받아 초기화한다.
 필터는 (FN, C, FH, FW)의 4차원 형상이다.
 여기서 FN은 필터 개수, C는 채널, FH는 필터 높이, FW는 필터 너비를 뜻한다.

 입력 데이터를 im2col로 전개하고 필터도 reshape을 사용해 2차원 배열로 전개한다.
 그리고 이렇게 전개한 두 행렬의 곱을 구한다.

 필터를 전개하는 부분은 각 필터 블록을 1줄로 펼쳐 세운다.
 이때 reshape의 두 번째 인수를 -1로 지정했는데, 이는 reshape이 제공하는 편의 기능이다.
 reshape에 -1을 지정하면 다차원 배열의 원소 수가 변화 후에도 똑같이 유지되도록 적절히 묶어준다.

 foward 구현의 마지막에서는 출력 데이터를 적절한 형상으로 바꿔준다.
 이때 넘파이의 transpose 함수를 사용하는데, 이는 다차원 배열의 축 순서를 바꿔주는 함수이다.
 아래 그림과 같이 인덱스를 지정하여 축의 순서를 변경한다.

 im2col로 전개한 덕분에 완전연결 계층의 Affine 계층과 거의 똑같이 구현할 수 있었다.

 합성곱 계층의 역전파를 구현할 때는 im2col을 역으로 처리해야 한다.
 col2im 함수를 이용하면 된다.
 col2im을 사용한다는 점을 제외하면 합성곱 계층의 역전파는 Affine 계층과 똑같으므로 생략하고 풀링 계층 구현하는 법을 살펴보자.

 ---

### 🔹 풀링 계층 구현하기

풀링 계층 구현도 합성곱 계층과 마찬가지로 im2col을 사용해 입력 데이터를 전개한다.
단, 풀링의 경우엔 채널 쪽이 독립적이라는 점이 합성곱 계층 때와 다르다.
구체적으로는 아래 그림과 같이 풀링 적용 영역을 채널마다 독립적으로 계산한다.

<img width="758" height="630" alt="image" src="https://github.com/user-attachments/assets/ff956fd7-7b5a-4a97-93db-1cd31c02c9dc" />

일단 이렇게 전개한 후, 전개한 행렬에서 행별 최댓값을 구하고 적절한 형상으로 성형한다.

<img width="767" height="372" alt="image" src="https://github.com/user-attachments/assets/89cdf050-5704-4fd5-96be-2687a5f72350" />

이상이 풀링 계층의 forward 처리 흐름이고, 이를 파이썬으로 구현하면 다음과 같다.

```python
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        # 전개 (1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        # 최댓값 (2)
        out = np.max(col, axis=1)

        # 성형 (3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        return out
```

풀링 계층 구현은 아래와 같이 세 단계로 진행된다.
 
1. 입력 데이터를 전개한다.
2. 행별 최댓값을 구한다.
3. 적절한 모양으로 성형한다.

풀링 계층의 forward 처리는 다음과 같다.
입력 데이터를 풀링하기 쉬운 형태로 전개하였고, 이후 backward 처리도 이전과 같으므로 생략한다.

### 🔹 CNN 구현하기

합성곱 계층과 풀링 계층을 구현했으니, 이 계층들을 조합하여 손글씨 숫자를 인식하는 CNN을 조립해보자.
여기에서는 다음과 같은 CNN을 구현한다.

"Convolution - ReLU - Pooling - Affine - ReLU - Affine - Softmax'
 
우선 SimpleConvNet 초기화 코드 먼저 구현해보자.

```python
class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28),
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        
        # conv layer 설정값 추출
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        # conv 출력 크기 계산
        conv_output_size = (input_size - filter_size + 2 * filter_pad) // filter_stride + 1

        # 풀링 후 출력 크기 계산
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)

        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)

        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

        self.last_layer = SoftmaxWithLoss()
```

여기에서는 초기화 인수로 주어진 합성곱 계층의 하이퍼파라미터를 딕셔너리에서 꺼낸다.
그리고 합성곱 계층의 출력 크기를 계산한다.
이어서 다음 코드는 가중치 매개변수를 초기화하는 부분이이다.

```python
self.params = {}
self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
self.params['b1'] = np.zeros(filter_num)

self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
self.params['b2'] = np.zeros(hidden_size)

self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
self.params['b3'] = np.zeros(output_size)
```

학습에 필요하 매개변수는 1번째 층의 합성곱 계층과 나머지 두 완전연결 계층의 가중치와 편향이다.
이 매개변수들은 인스턴스 변수 params 딕셔너리에 저장한다.

마지막으로 CNN을 구성하는 계층들을 생성한다.

```python
self.layers = OrderedDict()
self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],  conv_param['stride'], conv_param['pad'])
self.layers['Relu1'] = Relu()
self.layers['pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
self.layers['Relu2'] = Relu()
self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])

self.last_layer = SoftmaxWithLoss()
```

순서가 있는 딕셔너리인 layers에 계층들을 차례로 추가한다.
마지막 SoftmaxWIthLoss 계층만큼은 last_layer라는 별도 변수에 저장한다.

이제 추론을 수행하는 predict 메서드와 손실함수의 값을 구하는 loss 메서드를 구현하면 다음과 같다.



```python
def predict(self, x):
    for layer in self.layers.values():
        x = layer.forward(x)
    return x

def loss(self, x, t):
    y = self.predict(x)
    return self.last_layer.forward(y, t)
```

이 코드에서 인수 x는 입력 데이터, t는 정답 레이블이다.
추론을 수행하는 predict 메서드는 초기화 때 layers에 추가한 계층을 맨 앞에서부터 차례로 forward 메서드를 호출하며 그 결과를 다음 계층에 전달한다.

손실 함수를 구하는 loss 메서드는 predict 메서드의 결과를 인수로 마지막 층의 forward 메서드를 호출한다.
즉, 첫 계층부터 마지막 계층까지 foward를 처리한다.
 
이어서 오차역전파법으로 기울기를 구하는 구현은 다음과 같다.

```python
def gradient(self, x, t):
    # 손실 계산 (forward pass)
    self.loss(x, t)

    # 역전파 시작
    dout = 1
    dout = self.last_layer.backward(dout)

    layers = list(self.layers.values())
    layers.reverse()

    for layer in layers:
        dout = layer.backward(dout)

    # 결과 저장
    grads = {}
    grads['W1'] = self.layers['Conv1'].dW
    grads['b1'] = self.layers['Conv1'].db
    grads['W2'] = self.layers['Affine1'].dW
    grads['b2'] = self.layers['Affine1'].db
    grads['W3'] = self.layers['Affine2'].dW
    grads['b3'] = self.layers['Affine2'].db

    return grads
```

매개변수의 기울기는 오차역전파법으로 구한다.
이 과정은 순전파와 역전파를 반복한다.
지금까지 각 계층의 순전파와 역전파 기능을 제대로 구현했다면, 여기에서는 단지 그것들을 적절한 순서로 호출만 해주면 된다.
마지막으로 grads라는 딕셔너리 변수에 각 가중치 매개변수의 기울기를 저장한다.

이상이 SimpleConvNet의 구현 과정이다.

지금까지 살펴본 것처럼 합성곱 계층과 풀링 계층은 이미지 인식에 필수적인 모듈이다.
이미지라는 공간적인 형상에 담긴 특징을 CNN이 잘 파악하여 손글씨 숫자 인식에서 높은 정확도를 달성할 수 있었다.

### 🔹 CNN 시각화하기

CNN을 구성하는 합성곱 계층은 입력으로 받은 이미지 데이터에서 무엇을 보고 있는지 알아보도록 하자.
 
#### 1. 1번째 층의 가중치 시각화하기

이전에 MNIST 데이터셋으로 간단한 CNN 학습을 해보았는데, 그때 1번째 층의 합성곱 계층의 가중치는 그 형상이 (30, 1, 5, 5)였다.(필터 30개, 채널 1개, 5 X 5 크기).
 
필터의 크기가 5X5이고 채널이 1개라는 것은 이 필터를 1채널의 회색조 이미지로 시각화할 수 있다는 뜻이다.
그럼 합성곱 계층(1층) 필터를 이미지로 나타내보면 다음과 같다다.

<img width="780" height="276" alt="image" src="https://github.com/user-attachments/assets/aa4de598-8606-43ca-943e-73f140fd1d71" />

학습 전 필터는 무작위로 초기화되고 있어 흑백의 정도에 규칙성이 없다.
한편, 학습을 마친 필터는 규칙성이 있는 이미지화 되었다.
흰색에서 검은색으로 점차 변화하는 필터와 덩이리(블롭)가 진 필터 등, 규칙을 띄는 필터로 바뀌었다.

그림의 오른쪽같이 규칙성 있는 필터는 '무엇을 보고 있는' 걸까?
그것은 에지와 블롭(국소적으로 덩어리진 영역) 등을 보고 있다.
가령 왼쪽 절반이 흰색이고 오른쪽 절반이 검은색인 필터는 아래 그림과 같이 세로 방향의 에지에 반응하는 필터이다.

<img width="760" height="422" alt="image" src="https://github.com/user-attachments/assets/29f13bd6-a6c4-40a4-803a-1e24c0e507d3" />

위 그림은 학습된 필터 2개를 선택하여 입력 이미지에 합성곱 처리를 한 결과로, '필터 1'은 세로 에지에 반응하며 '필터 2'는 가로 에지에 반응하는 것을 알 수 있다.
 
이처럼 합성곱 계층의 필터는 에지나 블롭 등의 원시적인 정보를 추출할 수 있다.
이런 원시적인 정보가 뒷단 계층에 전달된다는 것이 앞에서 구현한 CNN에서 일어나는 일이다.

#### 2. 층 깊이에 따른 추출 정보 변화

앞의 결과는 1번째 층의 합성곱 계층을 대상으로 한 것이다.
1번째 층의 합성곱 계층에서는 에지나 블롭 등의 저수준 정보가 추추된다 치고, 그럼 겹겹이 쌓인 CNN의 각 계층에서는 어떤 정보가 추출될까?

계층이 깊어질수록 추출되는 정보(정확히는 강하게 반응하는 뉴런)는 더 추상화된다는 것을 알 수 있다.

아래 그림은 일반 사물 인식(자동차나 개 등)을 수행한 8층의 CNN이다.
이 네트워크 구조는 AlexNet이라 하는데, 합성곱 계층과 풀링 계층을 여러 겹 쌓고, 마지막으로 완전연결 계층을 거쳐 결과를 출력하는 구조이다.

블록으로 나타낸 것은 중간 데이터이며, 그 중간 데이터에 합성곱 연산을 연속해서 적용한다.

<img width="792" height="382" alt="image" src="https://github.com/user-attachments/assets/a1d93258-df23-40e3-910a-76723f58393d" />

딥러닝의 흥미로운 점은 위 그림과 같이 합성곱 계층을 여러 겹 쌓으면, 층이 깊어지면서 더 복잡하고 추상화된 정보가 추출된다는 것이다.
1번째 층은 에지와 블롭, 3번째 층은 텍스쳐, 5번째 층은 사물의 일부, 마지막 완전연결 계층은 사물의 클래스(개, 자동차 등)에 뉴런이 반응한다.
즉, 층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 '고급' 정보로 변화해간다.
다시 말하면 사물의 '의미'를 이해하도록 변화하는 것이이다.
 

### 🔹 대표적인 CNN

CNN 네트워크의 구성은 다양하다.
이번에는 그중에서도 특히 중요한 네트워크를 두 개 소개한다.
하나는 CNN의 원조인 LeNet이고, 다른 하나는 딥러닝이 주목받도록 이끈 AlexNet이다.

#### 1. LeNet

LeNet은 손글씨 숫자를 인식하는 네트워크로, 1988년에 제안되었다.
아래 그림과 같이 합성곱 계층과 풀링 계층(정확히는 단순히 '원소를 줄이기'만 하는 서브샘플링 계층)을 반복하고, 마지막으로 완전연결 계층을 거치면서 결과를 출력한다.

<img width="778" height="235" alt="image" src="https://github.com/user-attachments/assets/f94a70ea-0a04-4d87-9064-b691c1b5b466" />

LeNet과 '현재의 CNN'을 비교하면 몇 가지 면에서 차이가 있다.

첫 번째 차이는 활성화 함수이다.
LeNet은 시그모이드 함수를 사용하는 데 반해, 현재는 주로 ReLU를 사용힌다.
 
두 번째 차이는 서브 샘플링과 최대 풀링이다.
LeNet은 서브 샘플링을 하여 중간 데이터의 크기를 줄이지만 현재는 최대 풀링이 주류이다.

#### 2. AlexNet

LeNet과 비교해 훨씬 최근인 2012년에 발표된 AlexNet은 딥러닝 열풍을 일으키는 데 큰 역할을 했다.
그림에서 보듯 그 구성은 기본적으로 LeNet과 크게 다르지 않다.
 
<img width="777" height="273" alt="image" src="https://github.com/user-attachments/assets/36979046-4174-4001-ba45-692bb7e86bae" />
 
AlexNet은 합성곱 계층과 풀링 계층을 거듭하며 마지막으로 완전연결 계층을 거쳐 결과를 출력한다.
LeNet에서 큰 구조는 바뀌지 않았지만, AlexNet에서는 다음과 같은 변화를 주었다.
 
1. 활성화 함수로 ReLU를 이용한다.
2. LRN(Local Response Normalization)이라는 국소적 정규화를 실시하는 계층을 이용한다.
3. 드롭아웃을 사용한다.
---
