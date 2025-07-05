# 기계학습에서 수학의 역할 

- 수학은 목적함수를 정의하고, 목적함수가 최저가 되는 점을 찾아주는 최적화 이론 제공 
- 최적화 이론에 규제, 모멘텀, 학습률, 멈춤(stop)조건과 같은 제어를 추가하여 알고리즘 구축 
- 사람은 알고리즘을 설계하고 데이터를 수집함

![image](https://github.com/user-attachments/assets/e33e3958-d867-4785-8f8a-b1c90f9dbfd2)


# 벡터

- 샘플을 특징 벡터로 표현한다.
	ex) Iris 데이터에서 꽃 받침의 길이, 꽃 받침의 너비, 꽃잎의 길이, 꽃잎의 너비라는 4개의 특징이 각각 5.1, 3.5, 1.4, 0.2인 샘플

![image](https://github.com/user-attachments/assets/71f2947f-984a-433a-98da-6fcf19c7042f)


- 여러 개의 특징 벡터를 첨자로 구분한다.

![image](https://github.com/user-attachments/assets/d681c6db-2b9b-402e-8b3d-89a772ac272b)


# 행렬

**행렬(matrix)** 은 여러 개의 숫자(스칼라)를 **행(row)과 열(column)의 형태로 정렬한 2차원 구조**이며, 기계학습에서 데이터를 수학적으로 표현하고 연산하기 위해 필수적인 도구이다. 특히, **여러 개의 샘플을 특징 벡터(feature vector)로 표현**한 후 이들을 하나로 모아 구성한 것이 바로 행렬이다. 예를 들어, Iris 데이터셋처럼 각 샘플이 4개의 특징(꽃받침의 길이, 너비, 꽃잎의 길이, 너비)을 가진 경우, 150개의 샘플은 150×4 크기의 **설계 행렬(design matrix)** 로 표현될 수 있다.

ex) Iris 데이터에 있는 150개의 샘플을 설계 행렬 X로 표현

![image](https://github.com/user-attachments/assets/bc77e94a-1d26-4725-ade3-e45ae62a5f62)


행렬을 사용하면 **복잡한 연산을 간결한 수식**으로 표현할 수 있으며, **벡터의 내적**, **행렬 곱**, **전치(transpose)** 같은 기본 연산을 통해 다양한 알고리즘의 수학적 기초를 구성할 수 있다. 특히 **다변수 선형회귀, 신경망의 순전파/역전파, PCA 같은 차원 축소 기법**에서도 행렬 연산이 핵심 역할을 한다.

ex) 다항식의 행렬 표현

![image](https://github.com/user-attachments/assets/d392f7cf-77f6-4f30-8716-ae46febf52f5)


또한, 행렬은 **벡터의 집합**이라는 의미를 갖기 때문에 **고차원 벡터 공간**을 구성하거나, **선형 결합**을 통해 새로운 벡터를 생성하는 데 활용된다. 이처럼 행렬은 단순한 숫자 모음이 아니라, **데이터의 구조와 연산의 규칙을 수학적으로 표현하는 수단**으로, 기계학습과 딥러닝의 거의 모든 알고리즘에 필수적으로 사용된다.

 2x2의 A행렬과, B행렬을 각각 정의하고 출력하라.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("행렬 A:\n", A)
print("행렬 B:\n", B)
```

C행렬은 A행렬과 B행렬의 덧셈, D 행렬은 A에서 B를 뺀 행렬, E 행렬은 B에서 A를 뺀 행렬을 나타내는 행렬이다. 행렬의 덧셈 및 뺄셈에 대해 행렬을 정의하고, 출력하라.

```python
C = A + B
D = A - B
E = B - A

print("A + B:\n", C)
print("A - B:\n", D)
print("B - A:\n", E)

print("A @ B:\n", A @ B)
print("B @ A:\n", B @ A)
```

위 결과에서 D와 E는 서로 다른 값을 가지므로, **행렬의 뺄셈은 교환 법칙이 성립하지 않는다**는 것을 확인할 수 있다. 또한 **행렬 곱셈 역시 일반적으로 교환 법칙이 성립하지 않는다**. 
( A@B ≠ B@A )

2x2 행렬 x, y, z를 정의해주고, 분배 법칙( `A(B + C) = AB + AC` )이 성립하는지 확인해보자.

```python
X = np.array([[1, 2], [3, 4]])
Y = np.array([[2, 0], [1, 2]])
Z = np.array([[0, 1], [3, 1]])

left = X @ (Y + Z)

right = X @ Y + X @ Z

print("X @ (Y + Z):\n", left)
print("X @ Y + X @ Z:\n", right)
print("동일한가?:", np.allclose(left, right))
```

위에서 정의한 x,y,z 행렬을 그대로 사용하여, 결합법칙( `A(BC) = (AB)C` )이 성립하는지 확인해보자.

```python
left = X @ (Y @ Z)

right = (X @ Y) @ Z

print("X @ (Y @ Z):\n", left)
print("(X @ Y) @ Z:\n", right)
print("동일한가?:", np.allclose(left, right)) 
```

### 행렬 연산

- 행렬 곱셈에서 교환법칙은 성립하지 않는다 : `A@B ≠ B@A`
- 분배 법칙과 결합 법칙이 성립한다 : `A(B+C) = AB+AC` 이고, `A(BC) = (AB)C`

# 행렬 A의 전치 행렬 A^T

**전치행렬**이란, 행렬의 **행(row)과 열(column)을 뒤바꾼 것** 수학적으로는 `A^T` 또는 `A.T` 로 표현

![image](https://github.com/user-attachments/assets/9d73c71a-a3c2-41e0-ad7e-2d97f4443e7d)


아래 예제를 통해 전치 행렬을 구해보라.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])

A_T = A.T

print("원본 행렬 A:\n", A)
print("전치 행렬 A^T:\n", A_T)
```

- 이외 특수한 행렬들

![image](https://github.com/user-attachments/assets/b1dc533b-fba5-4f77-9fbb-2e0ef4358e75)


# 텐서

텐서(tensor)는 수학적으로 벡터, 행렬, 그 이상의 차원을 갖는 다차원 배열을 나타내는 개념
으로 다양한 형태의 데이터를 나타내는 데 사용되는 다차원 배열이다. 특히 기계 학습 분야
중 deep learning분야에서 데이터를 효율적으로 저장하고 연산하는 데 사용된다.

- 다차원 배열로 확장 및 연산: 숫자들로 구성된 데이터를 고려하면, 이 데이터를 1차원 배열로 혹은 2차원 배열 혹은 3차원 배열로 만들 수 있다. 이렇게 쉽게 데이터를 확장 및 조작을 할 수 있게 한 것이 텐서이다.
	예를 들면, 흑백이미지는 width와 height를 가진 2D 텐서(값은 흑백의 데이터 1 byte), 컬러이미지는 width, height, color를 가진 3D 텐서(컬러가 R,G,B이면 3개의 색상채널을 가짐. 즉, 각 색상 채널당 width와 height의 2D 데이터를 가짐)이다. 이러한 데이터들이 수학적 연산 및 텐서 연산을 통해 처리된다.

- 기계 학습에서의 활용: 텐서는 딥러닝에서 주요 데이터 구조이다.
	그림 혹은 영상들은 텐서로 변환하여 학습 모델에 입력으로 제공하며, 학습 모델의 가중치(weight)와 bias도 텐서로 표현된다. 텐서는 딥러닝에서 	데이터 및 학습 모델의 핵심 구성 요소이다.

### 텐서의 RANK 와 SHAPE

- Rank: 텐서의 차원의 수를 말한다.
	0차원 텐서는 스칼라(Scalar)로, 하나의 숫자로 구성된다.
	1차원 텐서는 벡터(Vector)로, 한 축만 있는 숫자의 배열이다.
	2차원 텐서는 행렬(Matrix)로, 숫자가 행과 열로 구성된 모양이다.
	3차원 텐서는 3차원 공간에 데이터를 나타내며, 2차원 텐서가 배열로 구성된 모양이다.
	이와 같이 계속해서 랭크가 증가할수록 차원이 더해진다.

- Shape: shape은 각 차원의 크기를 나타내는 튜플(tuple)이며, 텐서의 각 차원에 대한 크기를 순서대로 나타낸다. 즉, rank가 N이면 N-tuple로 구성된다.
	예를 들어, 3차원 텐서의 shape이 (2, 3, 4)이면, 이는 첫 번째 차원의 길이가 2, 두 번째 차원의 길이가 3, 세 번째 차원의 길이가 4인 배열을 의미한다. 여기서 길이는 요소의 개수를 의미한다. 따라서 shape은 텐서의 크기와 구조를 완전하게 설명해 주는 데이터이므로 shape을 알 면 텐서가 어떤 구조를 가지고 있는지 파악할 수 있다.

예를 들어, 다음은 텐서의 rank와 shape의 예시이다.
스칼라: rank 0, shape ()
벡터(길이가 5인): rank 1, shape (5)
행렬(3x3): rank 2, shape (3, 3)
3차원 배열(2x3x4): rank 3, shape (2, 3, 4)

numpy를 활용하여 Rank와, Shape, 그리고 기본적인 연산을 확인해보자.

- a.ndim은 Numpy 배열(텐서)의 차원의 수(Rank)를 확인하는 속성으로, number of dimensions의 약자이다.
- a.shape는 각 차원의 크기를 나타내는 속성이다.
- a.size는 전체 원소의 개수를 나타내는 속성이다.
- a.dtype은 데이터의 자료형을 나타내는 속성이다.

```python
import numpy as np

# 스칼라 (0차원 텐서)
scalar = np.array(7)
print("스칼라:", scalar)
print("Rank:", scalar.ndim)
print("Shape:", scalar.shape)

# 벡터 (1차원 텐서)
vector = np.array([1, 2, 3, 4, 5])
print("\n벡터:", vector)
print("Rank:", vector.ndim)
print("Shape:", vector.shape)

# 행렬 (2차원 텐서)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\n행렬:\n", matrix)
print("Rank:", matrix.ndim)
print("Shape:", matrix.shape)

# 3차원 텐서
tensor3d = np.array([
    [[1, 2, 3, 4],
     [5, 6, 7, 8],
     [9, 10, 11, 12]],
     
    [[13, 14, 15, 16],
     [17, 18, 19, 20],
     [21, 22, 23, 24]]
])
print("\n3차원 텐서:\n", tensor3d)
print("Rank:", tensor3d.ndim)
print("Shape:", tensor3d.shape)

```

## pip (Python Package Installer)

**pip(Python Package Installer)**는 Python에서 외부 라이브러리나 패키지를 설치할 수 있도록 도와주는 **표준 패키지 관리 도구**이다. Python 3.4 이상에서는 기본적으로 함께 설치되며, 개발자가 직접 수학, 과학, 웹, 인공지능 등 다양한 분야의 라이브러리를 추가할 때 사용된다. 예를 들어 `pip install numpy` 명령을 입력하면, Python 공식 패키지 저장소인 **PyPI(Python Package Index)**에서 `numpy`를 내려받아 설치한다.

pip은 명령어가 간단하고 가볍기 때문에 초보자부터 전문가까지 널리 사용되며, 대부분의 Python 환경(Jupyter Notebook, Google Colab 등)에서도 기본적으로 지원된다. 또한 `pip install`, `pip uninstall`, `pip list`, `pip show` 등의 명령어로 패키지의 설치, 제거, 정보 확인 등을 손쉽게 관리할 수 있다.

다만, pip은 패키지 간 의존성 문제를 완벽하게 해결하지는 못하므로, 여러 프로젝트를 동시에 관리할 때는 `venv`나 `virtualenv` 같은 **가상환경 도구**와 함께 사용하는 것이 좋다. 이로써 환경 간 충돌 없이 독립적인 개발 환경 구성이 가능하다.

다음 명령어를 통해 구글 코랩(Coolab)에서 pip를 통해 numpy를 설치해보자.
`!pip install <패키지명>`

![image](https://github.com/user-attachments/assets/d4047b0e-60e2-474b-9fac-ce12acd1c8b1)


### 주요 명령어

| 명령어                          | 설명             |
| ---------------------------- | -------------- |
| `!pip install`               | Colab에서 pip 설치 |
| `pip install 패키지명`           | 패키지 설치         |
| `pip install --upgrade 패키지명` | 패키지 업그레이드      |
| `pip uninstall 패키지명`         | 패키지 제거         |
| `pip list`                   | 설치된 패키지 목록 확인  |
| `pip show 패키지명`              | 특정 패키지 정보 확인   |

pip의 주요 장점은 설치 속도가 빠르고, PyPI에 등록된 수많은 패키지를 손쉽게 설치할 수 있다는 점이다. 특히 Colab, Jupyter 등 대부분의 환경에서 pip 명령어를 기본적으로 제공하므로 사용이 매우 편리하다.

반면, 단점으로는 복잡한 프로젝트에서 패키지 간 의존성 충돌이 발생할 가능성이 있다는 점이 있다. 따라서 보다 안전하고 안정적인 환경 구성을 위해서는 pip만 사용하는 것보다는 가상환경 도구와 함께 사용하는 것이 추천된다.

---

# 파일 시스템 이해 – 절대경로 vs 상대경로

## 경로(Path)란?

경로의 정의는 컴퓨터 내의 파일이나 폴더의 **위치 정보를 나타내는 문자열**을 말한다. 경로를 통해 운영체제나 프로그램은 원하는 파일에 접근할 수 있다. 컴퓨터에서 파일이나 폴더의 위치를 지정할 때 사용하는 경로(path)는 **절대 경로**와 **상대 경로**로 나눌 수 있다.  

**절대 경로**는 파일 시스템의 루트 디렉토리(`/`)부터 시작하여, 최종 파일 또는 폴더에 이르기까지의 전체 경로를 명시하는 방식이다. 예를 들어 `/home/user/project/data/sample.csv`와 같은 경로는 루트 디렉토리에서부터 정확하게 지정된 위치를 따라가야 하며, 현재 작업 중인 디렉토리와 관계없이 항상 같은 파일을 가리킨다. 따라서 절대 경로는 파일의 위치를 명확히 지정해야 하는 경우에 사용된다.
### 예시 (Linux/Mac 기준)

```text
/home/user/project/data/sample.csv
```

### 예시 (Windows 기준)

```text
C:\Users\user\project\data\sample.csv
```

반면, **상대 경로**는 현재 작업 중인 디렉토리(즉, 현재 위치)를 기준으로 하여 파일의 위치를 지정하는 방식이다. 예를 들어 `../data/sample.csv`라는 경로는 현재 위치에서 한 단계 위로 올라간 다음 `data` 폴더로 이동하여 `sample.csv` 파일을 찾는다는 의미이다. 상대 경로는 파일의 위치가 **작업 위치에 따라** 달라질 수 있으므로, 프로젝트 내부 파일 간의 연결이나 협업 시 유연하게 사용할 수 있다. 리눅스 상에서 pwd 명령어를 입력해서 현재 디렉토리의 위치를 확인할 수 있다.

| 기호                 | 의미                 |
| ------------------ | ------------------ |
| `.`                | 현재 디렉토리            |
| `..`               | 상위 디렉토리            |
| `./data/file.txt`  | 현재 폴더 아래 `data` 폴더 |
| `../data/file.txt` | 상위 폴더의 `data` 폴더   |

실제 프로그래밍이나 데이터 분석에서 경로를 지정할 때는 절대 경로가 명확하지만, 상대 경로는 이동성(포터블)과 관리 측면에서 유리하므로 목적에 따라 적절히 선택하여 사용해야 한다.

---

## 예제: 절대경로 vs 상대경로

```python
import os
import pandas as pd

# 현재 작업 디렉토리 확인
print("현재 위치:", os.getcwd())

# 절대경로로 파일 읽기
abs_path = "/content/sample_data/california_housing_train.csv"
df_abs = pd.read_csv(abs_path)
print("절대경로 읽기 성공")

# 상대경로로 파일 읽기
rel_path = "./sample_data/california_housing_train.csv"
df_rel = pd.read_csv(rel_path)
print("상대경로 읽기 성공")
```

> 실제 경로는 각자 실행 환경에 맞게 수정해야 한다.
> Colab에서는 `/content/`가 루트 경로이다.

| 항목 | 절대경로 | 상대경로 |
|------|-----------|-----------|
| 기준 위치 | 루트(`/`) 또는 드라이브 | 현재 작업 디렉토리 |
| 이동성 | 낮음 (시스템 고정) | 높음 (프로젝트 내부에 적합) |
| 예시 | `/home/user/data.csv` | `../data/data.csv` |
| 사용 용도 | 외부 파일, 시스템 경로 | 프로젝트 내부 파일 처리 |

---
### CSV란?

**CSV(Comma-Separated Values)** 파일은 데이터를 **쉼표(,)로 구분하여 저장하는 텍스트 기반의 파일 형식**이다. 각 줄은 하나의 데이터 행을 나타내며, 쉼표로 구분된 각 항목은 하나의 열에 해당한다. 예를 들어 `이름,나이,성별`이라는 헤더 아래 `철수,25,남`과 같은 데이터가 이어지며, 이는 마치 **엑셀의 표 형태**와 유사하다.

CSV는 구조가 단순하고 가독성이 뛰어나기 때문에 **엑셀, 데이터베이스, 통계 소프트웨어, 프로그래밍 언어 등 다양한 환경에서 널리 사용**된다. 특히, 용량이 가볍고 별도의 소프트웨어 없이 메모장 등으로 열어볼 수 있어 **데이터 공유와 저장에 매우 적합**하다.

Python의 pandas 라이브러리에서는 `pd.read_csv()`를 통해 CSV 파일을 손쉽게 불러올 수 있으며, `df.to_csv()`를 통해 다시 저장할 수 있다. 다만, 쉼표 외에도 탭(`\t`), 세미콜론(`;`) 등을 구분자로 사용하는 변형 형식도 존재하기 때문에 파일의 실제 구분자에 주의해야 한다.

요약하자면, CSV는 **간단하지만 범용적인 데이터 저장 형식**으로, 데이터 분석에서 가장 많이 사용되는 포맷 중 하나이다.

아래 예시 데이터 프레임을 만들어, student_scores.csv 파일을 저장하라.(`to_csv()`)
index=False를 사용해 Pandas의 인덱스를 저장하지 않도록 하라.

```python
import pandas as pd

data = {
    "이름": ["철수", "영희", "민수"],
    "국어": [85, 90, 78],
    "수학": [80, 95, 75]
}
df = pd.DataFrame(data)

df["평균"] = (df["국어"] + df["수학"]) / 2

df.to_csv("student_scores.csv", index=False)

print("CSV 파일 저장 완료!")
```

이후 `pwd` 혹은 os 패키지의 `os.getcwd()` 명령어를 사용하여 현재 디렉토리를 확인하여라.

```python
pwd
```

```python
import os
print("현재 작업 디렉토리:",os.getcwd())
```

이후 `pd.read_csv()` 를 사용하여 방금 저장한 파일을 DataFrame으로 복원하자.

```python
df_loaded=pd.read_csv("student_scores.csv")
print("CSV 파일에서 불러온 데이터 프레임:\n", df_loaded)
```

---
## pandas

**pandas**는 Python에서 데이터를 쉽고 효율적으로 다룰 수 있도록 도와주는 **오픈소스 데이터 분석 라이브러리**이다. 이름은 "panel data"에서 유래되었으며, 주로 **표 형태의 데이터(행과 열로 구성된 테이블)** 를 처리하는 데 최적화되어 있다. pandas는 특히 **엑셀처럼 데이터를 다룰 수 있으면서도**, **SQL처럼 필터링, 집계, 병합 등의 연산을 코드로 수행할 수 있는 점**에서 데이터 과학자, 분석가, 엔지니어들에게 필수 도구로 여겨진다.

pandas의 핵심 객체는 **`Series`와 `DataFrame`** 이다. `Series`는 1차원 데이터(리스트나 배열과 유사)를 다루며, `DataFrame`은 여러 개의 `Series`가 행과 열 구조로 결합된 2차원 데이터 구조이다. pandas는 CSV, Excel, SQL, JSON 등의 다양한 데이터 파일을 읽고 쓸 수 있으며, 누락값 처리, 필터링, 그룹화, 통계 연산, 시계열 분석 등 수많은 기능을 직관적인 문법으로 제공한다.

예를 들어 대용량의 엑셀 파일을 불러와 특정 조건에 따라 정리하거나, 웹에서 수집한 데이터를 통계적으로 요약하고 시각화 라이브러리로 넘겨주는 등의 작업을 효율적으로 수행할 수 있다. 또한 pandas는 `NumPy`와 `matplotlib` 같은 다른 과학 연산 도구들과 잘 호환되어 **Python 기반 데이터 분석 파이프라인의 중심 역할**을 한다.

| 기능 | 엑셀 | SQL | pandas |
|------|------|-----|--------|
| 대용량 처리 | ❌ 수십만 행에서 느려짐 | ✅ | ✅ (메모리 제한 내) |
| 복잡한 계산 | ❌ | ✅ | ✅ |
| 인터랙티브 탐색 | ✅ | ❌ | ✅ (Jupyter/Colab) |

---

### Series(1차원)

```python
import pandas as pd

s = pd.Series([10, 20, 30], index=['A', 'B', 'C'])
print(s)
print(s.index)   # Index(['A', 'B', 'C'], dtype='object')
print(s.values)  # array([10, 20, 30])
```

*Series는 값(value)과 인덱스(index)가 한 쌍으로 묶인 1차원 배열*을 말한다.  
넘파이 배열과 달리 **인덱스를 활용한 라벨 기반 접근**이 가능하다.

---

### DataFrame(2차원)
```python
data = {
    "이름": ["철수", "영희", "민수"],
    "국어": [90, 85, 78],
    "수학": [95, 88, 82]
}
df = pd.DataFrame(data)
df
```

`DataFrame`은 **여러 개의 Series가 같은 인덱스를 공유**하며 모인 **표 형태**의 자료구조이다.  
열(column)마다 자료형이 달라도 되므로 SQL 테이블과 유사한 개념이다.

### DataFrame은 numpy처럼 작동

- `DataFrame`은 내부적으로 **numpy 배열 기반**이기 때문에 **벡터 연산, 조건 연산, 브로드캐스팅**이 가능하다.

```python
import pandas as pd

df = pd.DataFrame({
    "국어": [90, 85, 78],
    "수학": [95, 88, 82]
}, index=["철수", "영희", "민수"])

# 열 단위 연산: 평균 점수 계산 (각 행에 대해)
df["평균"] = (df["국어"] + df["수학"]) / 2

# 브로드캐스팅: 수학 점수에 5점 가산
df["수학+보너스"] = df["수학"] + 5

print(df)
```

**출력 예시**:

```
      국어  수학   평균  수학+보너스
철수   90  95  92.5      100
영희   85  88  86.5       93
민수   78  82  80.0       87
```

- `numpy`처럼 연산자를 그대로 사용해 계산이 가능하며, 열 단위로 자동 정렬됨.
- `df["평균"] = np.mean(df, axis=1)`처럼 `numpy` 함수도 그대로 적용 가능.

> `DataFrame`은 `numpy`의 2차원 배열 + 라벨 기능을 추가한 구조로 이해하면 됩니다.

아래 표는 **pandas의 핵심 기능과 대표적인 메서드**들을 정리한 것이다. 데이터프레임을 다룰 때 자주 사용되는 기능들이며, 데이터 탐색부터 전처리, 분석, 저장까지 전 과정을 간결한 코드로 수행할 수 있다.

| 기능        | 메서드(예시)                                               | 설명                 |
| --------- | ----------------------------------------------------- | ------------------ |
| 행·열 미리 보기 | `df.head(3)`, `df.tail()`                             | 상·하단 일부 출력         |
| 행·열 선택    | `df['열']`, `df.loc[행라벨, '열']`, `df.iloc[행번호, 열번호]`    | 라벨/정수 기반 인덱싱       |
| 조건 필터링    | `df[df['점수'] > 80]`                                   | 불린 마스크             |
| 정렬        | `df.sort_values('점수', ascending=False)`               | 값 기준 정렬            |
| 통계 요약     | `df.describe()`, `df.mean()`                          | 기초 통계량             |
| 결측치 처리    | `df.isna()`, `df.dropna()`, `df.fillna(값)`            | NA 탐지/삭제/대체        |
| 그룹 분석     | `df.groupby('반')['점수'].mean()`                        | 그룹별 집계             |
| 데이터 병합    | `pd.concat([df1, df2])`, `pd.merge(df1, df2, on='키')` | 행·열 이어붙이기 / 조인     |
| 파일 입출력    | `pd.read_csv()`, `df.to_excel()`                      | CSV, Excel, JSON 등 |

```python
# 데이터 프레임을 이용한 panda 예제
import pandas as pd

# 샘플 데이터 생성 (학생 이름, 반, 국어, 수학 점수 포함)
data = {
    "이름": ["철수", "영희", "민수", "지민", "현우"],
    "반": ["A", "B", "A", "B", "A"],
    "국어": [85, 90, 78, 92, 88],
    "수학": [80, 95, 75, 89, 84]
}

# 데이터프레임 생성
df = pd.DataFrame(data)

# 평균 점수 컬럼 추가
df["평균"] = (df["국어"] + df["수학"]) / 2

# 출력
print("데이터프레임 예시:\n", df)
```

- `df.head()`와 `df.tail()`은 데이터의 상위 또는 하위 일부 행을 미리 보는 데 사용된다.

```python
# 1. 행·열 미리 보기
print("\n✅ 상위 3개 행 미리 보기:")
print(df.head(3))

print("\n✅ 하위 2개 행 미리 보기:")
print(df.tail(2))
```

- `df['열']`, `df.loc[]`, `df.iloc[]`은 열 또는 행을 선택할 때 사용되며, 라벨이나 숫자 인덱스를 기준으로 접근할 수 있다.

```python
# 2. 행·열 선택
print("\n✅ '수학' 열만 보기:")
print(df["수학"])

print("\n✅ 첫 번째 행, '국어' 점수 보기 (iloc):")
print(df.iloc[0, 2])  # 0번째 행, 2번째 열 (국어)

print("\n✅ '지민'의 평균 점수 보기 (loc):")
print(df.loc[df["이름"] == "지민", "평균"])
```

- `df[df['점수'] > 80]`처럼 조건식을 활용하면 원하는 조건에 맞는 데이터만 필터링할 수 있다.

```python
# 3. 조건 필터링
print("\n✅ 평균 점수가 85 이상인 학생:")
print(df[df["평균"] >= 85])
```

- 정렬은 `df.sort_values()`를 사용하며, 특정 열 기준으로 오름차순 또는 내림차순 정렬이 가능하다.

```python
# ✅ 4. 정렬
print("\n✅ 수학 점수 기준 내림차순 정렬:")
print(df.sort_values("수학", ascending=False))
```

- `df.describe()`는 기초 통계 정보를 요약해주고, `df.mean()`은 평균값을 계산하는 함수이다.

```python
# 5. 통계 요약
print("\n✅ describe() 통계 요약:")
print(df.describe())

print("\n✅ 평균 점수만 평균값 구하기:")
print(df["평균"].mean())
```

- 결측치 처리는 `df.isna()`, `df.dropna()`, `df.fillna()` 등을 통해 수행하며, NA 값을 탐지하거나 삭제하거나 다른 값으로 대체할 수 있다.

```python
# ✅ 6. 결측치 처리 (예시용 - 실제로 결측치는 없음)
print("\n✅ 결측치 탐지:")
print(df.isna())
```

- 그룹 분석에서는 `df.groupby()`를 사용하여 특정 열 기준으로 데이터를 집계하거나 평균을 낼 수 있다.

```python
# ✅ 7. 그룹 분석 (반별 평균)
print("\n✅ 반별 평균 점수:")
print(df.groupby("반")["평균"].mean())
```

- 데이터 병합은 `pd.concat()`을 통해 행이나 열 방향으로 이어붙일 수 있고, `pd.merge()`로 SQL처럼 조인 연산을 할 수 있다.

```python
# 8. 데이터 병합
print("\n✅ 다른 데이터프레임과 이어붙이기 예제:")
df2 = pd.DataFrame({
    "이름": ["정우"],
    "반": ["B"],
    "국어": [91],
    "수학": [87]
})
df2["평균"] = (df2["국어"] + df2["수학"]) / 2
df_combined = pd.concat([df, df2], ignore_index=True)
print(df_combined)
```

- `pd.read_csv()`, `df.to_excel()` 등은 외부 파일과 연동하여 데이터를 읽거나 저장할 때 사용된다.

```python
# ✅ 9. 파일 입출력
print("\n✅ CSV 파일로 저장 및 불러오기:")
df.to_csv("score.csv", index=False)
df_loaded = pd.read_csv("score.csv")
print("불러온 데이터프레임:\n", df_loaded)
```

이런 기능을 통해 pandas는 복잡한 데이터 처리 과정을 단순화하고, 효율적인 분석 환경을 제공한다.

아래 표는 **pandas를 이용한 외부 파일의 입출력 방법**을 정리한 것이다.  
pandas는 CSV, Excel, JSON 등 다양한 형식의 데이터를 **읽고 쓰는 기능**을 제공하며, 아래와 같이 간단한 함수 호출로 이를 처리할 수 있다.

| 형식 | 읽기 | 쓰기 |
|------|------|------|
| CSV | `pd.read_csv('data.csv')` | `df.to_csv('out.csv', index=False)` |
| Excel | `pd.read_excel('data.xlsx')` | `df.to_excel('out.xlsx', index=False)` |
| JSON | `pd.read_json('data.json')` | `df.to_json('out.json')` |
- **CSV 파일**은 `pd.read_csv()`로 읽고, `df.to_csv()`로 쓸 수 있다. 저장 시 `index=False` 옵션을 주면 인덱스 열 없이 저장된다.
- **Excel 파일**은 `pd.read_excel()`로 읽으며, 저장은 `df.to_excel()`을 사용한다. 역시 `index=False` 옵션을 통해 인덱스를 제외할 수 있다.
- **JSON 파일**은 `pd.read_json()`으로 읽고, `df.to_json()`으로 저장한다.

이처럼 pandas는 파일 형식에 관계없이 일관된 방식으로 데이터를 불러오고 저장할 수 있도록 도와주며, 데이터 분석 워크플로우에서 입출력을 손쉽게 처리할 수 있게 해준다.

---

### 과제 1) 3x3 행렬 연산에서 행렬의 법칙 확인

다음은 행렬 연산의 주요 법칙입니다:
- **교환 법칙** → A×B=B×A
- **분배 법칙** → A×(B+C)=A×B+A×C
- **결합 법칙** → A×(B×C) = (A×B)×C

크기가 3x3인 행렬 A,B,C를 정의하고, 행렬 연산에서 교환 법칙, 분배 법칙, 결합 법칙이 각각 성립하는지 확인하라. 
cf) 행렬의 곱은 A@B 형식으로 @를 사용

## 과제 2) pandas를 활용한 데이터 분석 실습

위의 강의자료 복습
