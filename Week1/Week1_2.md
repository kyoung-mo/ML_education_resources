# 객체지향언어
## 객체(Object)
객체는 자체의 속성과 행동을 함께 가지는 독립적인 단위로, 현실 세계의 사물이나 개념을 소프트웨어적으로 모델링한 것이다.

간단하게 설명하자면 **자체적인 변수와 함수를 갖는 새로운 속성의 코드**정도로 이해해도 좋다.
## 등장배경
소프트웨어가 점점 더 복잡해지고 크기가 커지면서 기존 절차지향언어로는 코드 관리와 유지보수가 어려워졌다. 개발자들은 코드 재사용 및 모듈화를 통해 빠르고 정확한 개발을 원했다. 

객체지향언어는 프로그램을 하나의 **'흐름'** 이란 개념이 아닌 다수의 **'객체'** 로 구성한다. 이는 현실 세계의 다양한 시스템과 유사하다. 

예를 들어, 건물의 위치와 기능, 관계도 등의 정보를 저장하여 지도로 시각화하는 프로그램을 개발한다고 가정하자. 만약 절차지향언어를 사용해서 하나의 흐름으로 프로그램을 개발한다면 코드가 너무 복잡해지고 새로운 건물이 생겨나는 등 프로그램을 수정할 때 매우 어려울 것이다.

객체지향언어라면 각 건물을 하나의 객체로 구현하고 다수의 객체를 동시에 원하는 만큼 사용하는 것으로 이를 구현할 수 있을 것이다. 만약 다른 지역의 동일한 프로그램을 사용하는 경우에도 해당 프로그램을 조금만 수정해서 구현할 수 있을 것이다.
## 클래스(class)
파이썬에서 클래스는 객체를 만들기 위한 설계도로 생각할 수 있다. C를 배운 학생이라면 **함수가 포함될 수 있는 구조체**정도로 생각해도 좋다. 파이썬에서는 정수, 실수, 리스트 등 모든 자료형이 객체고 클래스로 구현된다. 

```python
class A:
	B = 1
    def C(self):
	    print('C')
```

이렇게 `class` 키워드와 블록 구조로 클래스를 선언할 수 있다. Colab에서 코드를 작성하라.

```python
test = A()

print(test.B)
test.C()
print(B)
```

클래스를 사용하는 방식은 함수와 유사하다 `()`를 사용해 선언할 수 있다. 이때 클래스가 저장되는 `test`를 **인스턴스**라고 부른다.

클래스 내부의 변수와 함수를 사용하기 위해서는 `.`을 사용하여 접근할 수 있다. 그냥 `B`를 출력하면 오류가 발생한다. **클래스 내부의 변수는 오직 클래스 내부에서만 사용이 가능하다.**
### self
클래스 내부에서 만들어지는 함수를 **메소드**라고 한다. 이 메소드는 반드시 `self` 매개변수를 포함하여 작성되어야 하고, `self` 매개변수의 위치는 반드시 **첫번째**에 와야 한다.

`self` 매개변수는 메소드가 **자기 자신이 속한 클래스에 접근하기 위해 사용된다.** 해당 메소드 자체를 사용하기 위해선 기본적으로 클래스에 접근해야 하므로 `self` 매개변수는 반드시 작성되어야 한다.

```python
class A:
	B = 1
	
    def C(self):
		print(self.B)

	def D(self):
		self.B = self.B + 1

test = A()
test.C()
test.D()
test.C()
```

해당 코드를 실행해보자. 메소드 `C`,`D`는 각각 `B`를 사용하기 위해 `self.B` 구문을 사용했다. 이렇게 내부의 변수나 메소드를 사용하기 위해 `self`를 사용한다.
## 매직 메소드
파이썬에서는 클래스를 만들고 인스턴스를 선언하고 메소드를 호출하여 기능을 수행한다. **매직 메소드**란 **코드로 직접 호출하지 않고 특별한 상황에서 자동으로 호출되는 메소드를 말한다.**
### 생성자
생성자는 인스턴스를 선언할 때 자동으로 호출되는 메소드다. `__init__()` 이란 이름으로 메소드를 작성하면 이 메소드는 인스턴스 선언 시 자동으로 사용된다.

```python
class A:
	
    def __init__(self):
		print('인스턴스 선언 완료')

test = A()
```

생성자에 **매개변수를 추가해서 인스턴스 선언 시 인자를 받게 구현할 수도 있다.**

```python
class A:
	
    def __init__(self, a, b):
		print('인스턴스 선언 완료')
		self.a = a
		self.b = b

	def B(self):
		print(self.a)
		print(self.b)

test = A(1,2)
test.B()
```

생성자는 인스턴스 선언 시 값을 원하는 대로 셋팅하고 인스턴스마다 생성 시 새로운 값을 인자로 받을 수 있기 때문에 매우 자주 사용되는 매직 메소드다.

생성자말고도 매우 다양한 매직 메소드가 있다.
- `__del__(self)`
- `__str__(self)`
- `__len__(self)`
- `__getitem__(self)`
## 상속
개발자가 어떤 클래스 A를 만들었다고 가정하자. 추가로 클래스 B를 만들고자 할 때 **클래스 B가 A에서 사용한 메소드 및 변수를 동일하게 사용하고자 한다면 '상속'으로 쉽게 구현할 수 있다.** 

상속이란 클래스를 만들 때 계층적인 구조로 구현할 수 있게 해주는 문법이다.

```python
class person:
    def __init__(self, a, b):
        self.age = a
        self.country = b

    def old(self):
        self.age += 1

class programmer(person):
    def __init__(self, a, b): 
        super().__init__(a, b) 
        print(self.age)
        print(self.country)

    def prin(self):
        super().old()
        print(self.age)
        print(self.country)

test = programmer(26, "korea")
test.old()
test.prin()
```

위 코드에서 `old(self)` 메소드를 하위 클래스에서 수정하여 사용했다. 이렇게 상속 해주는 클래스를 **부모 클래스**라고 하고 상속 받는 클래스를 **자식 클래스**라고 한다. 

`super()`은 자식 클래스에서 부모 클래스의 메소드를 호출할 수 있도록 해주는 내장 함수다.

**자식 클래스는 다수의 부모클래스를 가질 수도 있다.**
# 패키지와 라이브러리
파이썬에서 클래스와 상속을 활용해 다양한 객체와 기능을 모듈화할 수 있게 되면서, 개발자들 간에 코드의 재사용과 공유가 활발해졌다.

그러나 하나의 파이썬 파일 `.py` 에 모든 클래스와 함수를 작성해야 하는 것은 매우 비효율적이고 안전하지 않다.

용도에 맞게 개발된 함수와 클래스가 작성된 파이썬 파일을 **모듈**이라고 하고, 다수의 모듈로 구성된 폴더 구조를 **패키지**라고 한다. 패키지와 모듈을 모아놓은 것을 **라이브러리**라고 한다.

라이브러리
├── 패키지 (폴더)
│   ├── 서브패키지
│   │   ├── 모듈 (.py 파일)
│   │   └── ...
│   └── 모듈
└── 단일 모듈 

**라이브러리 ⊃ 패키지 ⊃ 모듈 ⊃ 함수/클래스**

파이썬은 `pip`나 `conda` 등 추가 기능으로 원하는 패키지 및 라이브러리를 인터넷에서 다운로드하여 사용할 수 있다. 자세한 내용은 추후에 다시 설명하도록 하자.

다양한 기능을 갖춘 라이브러리 생태계가 있기 때문에 파이썬은 다양한 분야에서 자주 사용된다.
# Numpy
이번 시간에는 Colab 이미 설치된 `numpy` 라이브러리를 사용해서 어떤 기능을 사용할 수 있는지 확인한다. 개인적인 개발환경이 있다면 따로 `numpy`를 설치해야한다. 

`numpy`는 다차원 배열을 효율적으로 처리하고 벡터화 연산, 선형 대수, 난수 생성, 푸리에 변환등 다양한 수학/과학 계산에 주로 사용되는 고성능 수치 계산을 위한 라이브러리다.

개발자는 `import` 구문을 이용해 **코드 밖에 있는 파이썬 파일의 함수와 클래스를 사용할 수 있다.**

```python
import A
```

`numpy`를 불러올 때는 주로 `as` 를 사용하여 `np` 로 줄여 사용한다.

```python
import numpy as np
```
## np.array
`array`는 `numpy`의 가장 기본적인 연산 단위인 배열 클래스다. 

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])
print(a)
print(b)
print(a.shape)
print(b.shape)
print(a.dtype)
print(b.dtype)
print(a.ndim)
print(b.ndim)
```

`array` 인스턴스 선언 시 다차원의 리스트를 인자로 사용하여 선언가능하다. 해당 클래스의 기본적인 변수는 `shape`, `dtype`, `ndim` 등이 있다. 
- `shape` : 배열의 크기를 튜플로 저장.
- `dtype` : 배열 요소의 자료형을 저장. `array`의 모든 요소는 같은 자료형으로 통일됨.
- `ndim` : 배열의 차원 수를 저장.

리스트말고도 배열을 원하는대로 생성하는 함수가 존재한다.

```python
zero = np.zeros((2, 3))
one = np.ones((3, 3))
eye = np.eye(4)
rag = np.arange(0, 10, 2)
lin = np.linspace(0, 1, 5)
print(zero)
print(one)
print(eye)
print(rag)
print(lin)
```

- `zeros` : 인자로 받은 크기만큼 0 요소로 찬 배열을 생성.
- `ones` : 인자로 받은 크기만큼 1 요소로 찬 배열을 생성.
- `eye` : 인자로 받은 크기만큼 단위 행렬 생성.
- `arange` : 등차 수열의 정수 배열을 생성.
- `linspace` : 등간격으로 나눈 실수 배열 생성.
## 인덱스
`array`에서도 인덱스가 존재한다.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr[0, 1])
print(arr[:, 1])
```

리스트나 튜플처럼 기본 Python 자료형에서는 2차원 이상 배열의 경우 `[a][b][c]` 처럼 인덱스를 지정해야 요소의 값에 접근할 수 있었지만, `array`의 경우 `[a, b, c]` 의 형태로 인덱스 지정이 가능하다. 또한, 기본 자료형처럼 `:`를 활용한 슬라이싱이 가능하다.
## 행렬 연산
`numpy`의 가장 강력한 기능은 다차원 행렬 연산이다. 리스트만으로 행렬 곱셈을 구현한다고 가정해보자.

```python
def matmul(A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):  
                result[i][j] += A[i][k] * B[k][j]
    
    return result

A = [[1, 2],
     [3, 4]]

B = [[5, 6],
     [7, 8]]

C = matmul(A, B)
print(C)
```

행렬 곱셈만을 위한 함수를 정의해야 하고, 3차원 이상의 배열은 더욱 복잡할 것이다. 그러나 `numpy`는 자체적인 고속 행렬 연산을 지원한다.

```python
import numpy as np

A = np.array([[1, 2],
              [3, 4]])
B = np.array([[5, 6],
              [7, 8]])

print(A@B)
```

행렬 곱 뿐만 아니라 덧셈, 뺄셈, 요소별 곱셈 등 다양한 연산을 간단한 연산자로 구현 가능하다. 다차원 Tensor의 경우에도 동일한 연산을 지원한다.

```python
print(A+B)
print(A-B)
print(A*B)
print(A/B)
```
## 형태 조작
`array`의 내부 요소를 변경하지 않고 형태만 변경하는 연산도 지원한다.

```python
A = np.array([[1, 2],
              [3, 4],
              [5, 6]])

print(A)
print(A.reshape(2,3))
print(A.flatten())
print(A.T)
```

- `.reshape(X)` : `X`는 튜플. 현재 `array`의 `shape`를 지정한 `X`의 형태로 변환함. 단, 반드시 `array`의 크기와 변환하고자하는 `X`의 크기가 동일해야 함.
- `.flatten()` : 다차원의 `array`를 **1차원 배열로 길게 늘어뜨려 변환함.**
- `.T` : `array`의 전치행렬을 반환함.
## 배열 결합
다수의 `array`를 결합하여 새로운 `array`를 만들어야 하는 상황에서는 다음과 같은 함수를 사용할 수 있다.
### concatenate()

```python
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

result = np.concatenate((a, b, c))
print(result)
print(result.shape)

a = np.array([[1, 2], [3,4]])
b = np.array([[5, 6], [7,8]])

result1 = np.concatenate((a, b), axis=0)
print(result1)
print(result1.shape)
result2 = np.concatenate((a, b), axis=1)
print(result2)
print(result2.shape)
```

`concatenate`는 다수의 `array` 를 결합할 때 사용 가능한 함수다. 결합하고자 하는 `array`를 튜플로 묶어 인자로 전달하면 결합된 새로운 `array`를 반환한다. 추가로 `axis` 인자를 가지고 있는데, 이는 **결합하고자 하는 차원 축**을 의미하며 기본값은 `0`으로 설정되어 있다. 

2차원 행렬의 경우 `axis`를 최대 1까지, 3차원의 경우 최대 2까지 설정하여 **어느 방향으로 두 배열을 결합시킬지 결정**할 수 있다.
### stack()
```python
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])

result = np.stack((a, b, c))
print(result)
print(result.shape)

a = np.array([[1, 2], [3,4]])
b = np.array([[5, 6], [7,8]])

result1 = np.stack((a, b), axis=0)
print(result1)
print(result1.shape)
result2 = np.stack((a, b), axis=1)
print(result2)
print(result2.shape)
```

`concatenate`의 경우 2차원 행렬을 결합하면 2차원 행렬이 반환되었다. `stack`은 **새로운 차원을 추가하여** `array`를 결합하는 방식이다. 1차원 배열을 결합하면 2차원 행렬을 반환하고, 2차원 행렬을 결합하면 3차원 행렬을 반환한다. 여기서도 `axis`는 결합하고자 하는 차원의 방향을 결정한다.
![[그림1.png]]
두 결합 방식은 각각 새로운 차원을 생성하느냐 아니냐의 차이가 있다. 
## 인덱스 추출
필요한 경우 `array`에서 **요소들을 확인하여 조건을 만족하는 요소의 인덱스를 추출**해야 하는 경우가 생길 수 있다.

```python
a = np.array([10, 25, 17, 30])
idxs = np.where(a > 20)

print(idxs)     
print(a[idxs])   
print(np.argmax(a)) 
print(np.argmin(a))
```

- `.where(X)` : `X`는 `array`와 동일한 크기의 `boolean` 배열만 가능. 해당 조건을 만족하는 `array` 인덱스를 반환함.
- `.argmax()` : `array` 중 최대값을 가진 요소의 인덱스를 반환.
- `.argmin()` : `array` 중 최솟값을 가진 요소의 인덱스를 반환.
# 예제
```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"안녕하세요, 저는 {self.name}이고 {self.age}살입니다.")

    def birthday(self):
        self.age += 1
        print(f"{self.name}의 생일입니다! 이제 {self.age}살이 되었습니다.")

class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)  
        self.student_id = student_id
        self.scores = {}  

    def add_score(self, subject, score):
        self.scores[subject] = score
        print(f"{subject} 점수 {score}점이 추가되었습니다.")

    def get_average(self):
        if not self.scores:
            return 0
        return sum(self.scores.values()) / len(self.scores)

    def show_info(self):
        self.introduce()  
        print(f"학번: {self.student_id}")
        print(f"과목별 점수: {self.scores}")
        print(f"평균 점수: {self.get_average():.2f}")


person = Person("김영희", 30)
person.introduce()
person.birthday()
print('-------------------------------------')

student = Student("이철수", 17, "2025A001")
student.show_info()
print('-------------------------------------')

student.add_score("수학", 85)
student.add_score("영어", 92)
student.add_score("과학", 78)
print('-------------------------------------')
student.show_info()
```

첫번째 예제는 이름과 나이 변수, 자기소개와 생일 메소드를 가진 부모 클래스로부터 성적 산출 기능이 추가된 자식 클래스를 작성하는 예제다. 

```python
import numpy as np

def is_row_independent(A):
    A = np.array(A, dtype=float)
    n_rows = A.shape[0]
    row = 0
    for col in range(A.shape[1]):
        for r in range(row, n_rows):
            if A[r][col] != 0:
                A[[row, r]] = A[[r, row]]
                break
        else:
            continue
        for r in range(row + 1, n_rows):
            ratio = A[r][col] / A[row][col]
            A[r] = A[r] - ratio * A[row]
        row += 1
    return row == n_rows

A1 = np.array([[1, 2, 3],
               [0, 1, 4],
               [0, 0, 5]])

A2 = np.array([[1, 2, 3],
               [2, 4, 6],
               [3, 6, 9]])

print(is_row_independent(A1)) 
print(is_row_independent(A2))
```

두번째 예제는 행렬 입력에 대해 해당 행렬이 **선형 독립**인지 확인하는 예제다. 선형 독립이란, **어떠한 행도 다른 행들의 선형 결합으로 구할 수 없는 상태**를 말한다.

**가우스 소거법**을 사용하여 해당 행렬이 선형 독립인지 판단한다.
# 과제 1-2
## 1번 과제
`class` 와 상속을 사용하여 건물 객체들을 생성하는 코드를 작성하라.
- 각 건물마다 고유한 기능이 존재해야 한다. Ex) 병원 - 치료, 은행 - 대출 등
- 건물의 고유한 기능은 메소드와 클래스 변수를 사용하여 구현하라.
## 2번 과제
https://numpy.org/doc/2.3/reference/index.html 에서 `numpy`의 다양한 함수 및 기능을 확인할 수 있다. 이 중 1개의 함수 또는 메소드를 찾아 사용하라.
