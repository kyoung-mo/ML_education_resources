# 저번 주차 과제
---

# 과제 1) 다음 조건을 기반으로 Convolution 연산 후 출력 feature map의 크기를 계산하시오.

입력 이미지의 크기가 28×28 이고, 필터의 크기는 5×5, 스트라이드는 2, 패딩은 1일 때, 
하나의 필터를 적용한 후의 출력 feature map의 크기는 얼마인가?

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

### 🔹 Floor 함수 (바닥 함수) 설명
- 기호: `⌊ ⌋`

수학에서 `⌊x⌋`는 **floor 함수** 또는 **바닥 함수**라고 하며,  
**소수점 이하를 버리고 가장 가까운 정수로 내림**한 값을 의미한다.

- 예시
  - ⌊5.7⌋ = 5  
  - ⌊12.5⌋ = 12  
  - ⌊25 / 2⌋ = ⌊12.5⌋ = 12
 
### 🔹 과제 1 계산:

$$
\left\lfloor \frac{28 - 5 + 2 \times 1}{2} \right\rfloor + 1
= \left\lfloor \frac{25}{2} \right\rfloor + 1
= \left\lfloor 12.5 \right\rfloor + 1
= 12 + 1 = 13
$$

따라서 출력 크기는 **13 × 13**이 된다.

---
# 저번 주차 복습

### 🔹 합성곱 신경망(convolution neural network:CNN)

- 합성곱 계층과 풀링 계층으로 구성

### 🔹 완전연결층 (fully connected = 전결합층 = Affine 계층)

- Affine 계층을 사용할 때, 만약 층이 5개인 완전연결 신경망은 아래와 같이 구현된다.
- 완전 연결 신경망은 Affine 계층 뒤에 활성화 함수를 갖는 ReLU 계층 (혹은 Sigmoid 계층)이 이어진다.
- 아래 그림은 Affine-ReLU 조합이 쌓였고, 마지막 5번째 층은 Affine 계층에 이어 소프트맥스 계층에서 최종 결과(확률)를 출력한다.

<img width="904" height="202" alt="image" src="https://github.com/user-attachments/assets/aff2e7c4-fc9c-445f-bf21-e755ef2ae3d9" />


### 🔹 합성곱 신경망(convolution neural network: CNN)

- 아래의 그림과 같이 CNN에는 새로운 합성곱 계층(Conv)와 풀링 계층(Pooling)이 추가된다.
- Affine-ReLU의 연결 -> Conv-ReLU-Pooling 으로 바뀐다.
- CNN에서는 Affine-ReLU의 구성이 가능하며, 마지막 출력 계층에서는 Affine-Softmax를 그대로 사용할 수 있다.

<img width="901" height="180" alt="image" src="https://github.com/user-attachments/assets/7a001683-33f2-4c91-b1eb-ef99d9e8f887" />


### 🔹 합성곱 계층

- CNN에서는 패딩, 스트라이드 등 CNN 고유의 용어가 등장.
- 각 계층 사이에는 3차원 데이터같이 입체적인 데이터가 흐른다는 점에서 완전연결 신경망과 다르다.
- 합성곱 계층의 입출력 데이터를 특징맵(feature map)이라고 한다.
  - 입력 특징 맵(input feature map)
  - 출력 특징 맵(output feature map)
 
### 🔹 완전연결 계층의 문제점

- 데이터의 형상이 무시된다는 문제점이 있다.
- 형상을 무시하고 모든 입력 데이터를 동등한 뉴런으로 취급하여 형상에 담긴 정보를 살릴 수 없다.

- 하지만 합성곱 계층은 형상을 유지한다.
- 이미지도 3차원 데이터로 입력을 받고, 마찬가지로 다음 계층에서도 3차원 데이터로 전달한다. (이미지 처리에 특화)

- 완전 연결(Fully-Connected)은 1줄로 세운 데이터를 Affine 계층에 입력하여 데이터의 형상을 무시했다면, 이미지처럼 3차원 형태를 합성곱 계층으로 사용하여 위치 정보를 전달하고 형상을 유지할 수 있다.

### 🔹 합성곱의 연산

- 합성곱 계층에서의 합성곱 연산을 처리한다.
- 이미지 처리의 필터 연산에 해당한다.
- 필터의 형상을 높이와 너비로 표현하며, 필터를 커널이라 부르기도 한다.
- 합성곱 연산은 필터의 윈도우(window)를 일정한 간격으로 이동해가며 입력 데이터에 적용한다.
- 합성곱의 연산은 단일 곱셈-누산(Fused Multiply-Add, FMA라고 표현)

<img width="596" height="193" alt="image" src="https://github.com/user-attachments/assets/0e1827db-fa08-4316-8354-4d9338118f89" />


- 합성곱 연산 예시

<img width="751" height="942" alt="image" src="https://github.com/user-attachments/assets/44448ecd-6968-4a94-aab7-0a3fac58c60e" />


---

- 완전연결 신경망에는 가중치와 편향이 존재한다.
- CNN에서는 필터의 매개변수가 그동안의 가중치에 해당하며, CNN에서도 편향이 존재한다.
- 편향은 항상 하나(1x1)만 존재한다.

<img width="922" height="200" alt="image" src="https://github.com/user-attachments/assets/e0ab773b-c565-4591-9d65-3f2624bccb49" />


필터를 적용한 원소에 고정값(편향)을 모두 더해준다.

### 🔹 패딩(Padding)

- 합성곱 연산을 수행전에 입력 데이터 주변을 특정 값(주로 0을 사용)으로 채운다.
- 패딩은 주로 출력 크기를 조정할 목적으로 사용한다.
- 합성곱 연산을 거칠때마다 크기가 작아지면 어느 시점에서는 출력의 크기가 1이 되어버릴수 있으므로 이를 막기 위해 패딩을 사용한다.

<img width="703" height="308" alt="image" src="https://github.com/user-attachments/assets/1896c28f-74c8-476b-bb26-e6fb9d69701a" />


### 🔹 스트라이드(Stride)

- 필터를 적용하는 위치의 간격을 스트라이드라고 한다.
- 예를 들어 스트라이드를 2로 설정하면 필터를 적용하는 윈도우가 두 칸씩 이동한다.

<img width="764" height="496" alt="image" src="https://github.com/user-attachments/assets/bb551118-1577-48f1-9a21-7363b23760ea" />



### 🔹 출력의 크기(Output Size)

- 입력 크(H, W), 필터 크기(FH, FW), 출력 크기 (OH, OW), 패팅 P, 스트라이트 S
- 단, 정수로 나눠떨어지는 값이어야 한다는 점을 주의해야 한다. 정수가 아니면 오류를 내는 등의 대응을 해줘야 한다. (가장 가까운 정수로 반올림하는 등, 특별히 에러를 내지 않고 진행하도록 구현하는 경우도 있다)

<img width="463" height="202" alt="image" src="https://github.com/user-attachments/assets/c8d1b76a-8cca-47eb-9d3c-7f2e9dc57319" />


--- 
예시)

<img width="990" height="373" alt="image" src="https://github.com/user-attachments/assets/2d958783-9b85-4a27-98a8-6913382d9785" />


  1. 입력 크기:4x4, 필터 크기:3x3, 패딩:0, 스트라이드:1 -> 출력 크기=2x2
  2. 입력 크기:5x5, 필터 크기:4x4, 패딩:2, 스트라이드:1 -> 출력 크기=6x6
  3. 입력 크기:6x6, 필터 크기:3x3, 패딩:1, 스트라이드:2 -> 출력 크기=3x3


---
# 이번 주차 진도
---


### 🔹 3차원 데이터의 합성곱 연산

- 이미지만 해도 세로, 가로에 더해서 채널까지 고려한 3차원 데이터이다. 2차원 일때와 비교하면, 길이 방향(채널 방향)으로 특징 맵(feature map)이 늘어난다.
- 채널 쪽으로 특징 맵이 여러 개 있다면 입력 데이터와 필터의 합성곱 연산을 채널마다 수행하고, 그 결과를 더해서 하나의 출력을 얻는다.
- 3차원의 합성곱 연산에서 주의할 점은 입력데이터의 채널수와 필터의 채널수가 같아야 한다.

<img width="685" height="937" alt="image" src="https://github.com/user-attachments/assets/895e41f2-8461-44fe-97f3-2c9643f517e5" />


### 🔹 블록으로 생각하기

- 3차원 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각하면 된다.
- 입력데이터(C, H, W), 필터(C, FH, FW) = 출력데이터(1,OH,OW)
  - C : Channel 의미
  - FH : Filter Height
  - FW : filter Width

<img width="746" height="235" alt="image" src="https://github.com/user-attachments/assets/f82f8233-28f4-4930-b89c-9cb5ce5baee2" />


---

- 그렇다면 합성 곱 연산의 출력으로 다수의 채널을 보내려면?
  - ->다수의 필터를 사용한다.

- 사용된 필터의 개수만큼 출력 데이터의 채널수 출력

<img width="767" height="448" alt="image" src="https://github.com/user-attachments/assets/d73ab8b6-fd1d-42db-b368-9888bcca80c4" />


- 입력데이터 (C, H, W) * 필터 (C, FH, FW, FN) = 출력데이터(FN, OH, OW)
- 여기서 필터를 FN개 적용하면 출력 맵도 FN개 생성된다. 

- 위의 그림에서 보는 것과 같이 합성곱 연산에서는 필터의 수도 고려해야 한다. 
- 그런 이유로 필터의 가중치 데이터는 4차원 데이터이며, (출력 채널수, 입력 채널수, 높이, 너비) 순으로 순으로 쓰인다.

ex) 채널수 3, 크기 5x5 필터 20개, (20, 3, 5, 5)

 

합성곱 연산에도(완전 연결 계층과 마찬가지로) 편향이 쓰인다.
입력데이터 (C, H, W) * 필터 (C, FH, FW, FN) => 출력데이터(FN, OH, OW) + 편향 (FN, 1, 1) => (FN, OH, OW)
형상이 다른 블록의 덧셈은 넘파이의 브로드캐스트 기능으로 쉽게 구현이 가능하다.

<img width="798" height="319" alt="image" src="https://github.com/user-attachments/assets/48dd2902-f71e-4edf-aea9-ec3ff3ed7264" />



### 🔹 배치 처리

- 신경망 처리에서는 입력 데이터를 한 덩어리로 묶어 배치를 처리했다.
- 완전 연결 신경망을 구현하면서는 이 방식을 지원하여 처리 효율을 높이고, 미니배치 방식의 학습도 지원하였다.
- 합성곱 연산도 마찬가지로 배치 처리를 지원하며, 각 계층을 흐르느 데이터의 차원을 하나 늘려 4차원 데이터로 저장한다.

<img width="912" height="310" alt="image" src="https://github.com/user-attachments/assets/4521ba25-6f61-4abd-8a2e-f2756fac8d86" />


- 이처럼 데이터는 4차원 형상을 가진 채 각 계층을 타고 흐른다.
- 여기에서 주의할 점으로는 신경망에 4차원 데이터가 하나 흐를 때마다 데이터 N개 대한 합성곱 연산이 이뤄진다는 것이다.
  - 즉, N 회 분의 처리를 한 번에 수행하는 것이다

### 🔹 풀링 계층

- 풀링은 세로 가로 방향 공간을 줄이는 연산이다.
- 최대 풀링(Max Pooling)은 대상 영역에서 최댓값을 취하는 연산인 반면, 평균 풀링(Average Pooling)은 대상 연역에서 평균을 계산한다.
- 이미지 인식 분야에서는 주로 최대 풀링을 사용한다.

<img width="880" height="354" alt="image" src="https://github.com/user-attachments/assets/a209a286-7320-4c67-9e37-eb5a534789b1" />


- 풀링 계층의 특징(3)
  - 1. 학습해야 할 매개변수가 없다.
       - 풀링은 대상 영역에서 최댓값이나 평균을 취하는 명확한 처리이므로 특별히 학습할 것이 없다.
  - 2. 채널 수가 변하지 않는다.
       - 풀링 연산은 입력 데이터의 채널 수 그대로 출력데이터로 내보낸다.
       - 채널마다 독립적으로 계산하기 때문이다.
  - 3. 입력의 변화에 영향을 적게 받는다. (강건하다)
       - 입력데이터가 조금 변해도 풀링의 결과는 잘 변하지 않는다.
 
<img width="850" height="215" alt="image" src="https://github.com/user-attachments/assets/82b27ba9-a419-4d3d-82d5-fe702c6cf4a7" />




---
