
---
# Jetson Orin Nano 개발 환경 세팅
---

<img width="892" height="432" alt="image" src="https://github.com/user-attachments/assets/0eca7ff4-c120-4644-aced-4a236b4caea1" />

### 🔹 Jetson Orin Nano란?
- NVIDIA에서 개발한 엣지 AI 및 임베디드 시스템을 위한 소형 컴퓨팅 보드  
- 고성능 GPU(Ampere 아키텍처) 및 ARM 기반 CPU 탑재  
- ROS2, TensorRT, CUDA, cuDNN 등 다양한 AI 및 로봇 개발 프레임워크 지원  
- JetPack SDK를 통해 Ubuntu 기반 개발 환경 제공  

---
# PuTTY
---

<img width="755" height="447" alt="image" src="https://github.com/user-attachments/assets/de3be944-4104-4fe9-a247-2b6c8311ba53" />

### 🔹 PuTTY 소개
- Windows 및 기타 운영 체제에서 SSH, Telnet, Serial 연결을 지원하는 소프트웨어  
- Host PC와 Jetson Orin Nano 간 유선 통신 가능  
- 경량화된 무료 오픈소스 프로그램으로 손쉽게 설치 및 사용 가능  

### 🔹 설치 방법
- [공식 사이트](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)에서 다운로드 가능  
- Windows Installer(x86/x64) 또는 Unix용 소스 아카이브 제공  
- 설치 후 바로 SSH 접속 설정을 통해 Jetson Orin Nano 연결 가능  

<img width="1263" height="496" alt="image" src="https://github.com/user-attachments/assets/91cf0214-47f0-4e93-b8ad-99d2e2e7cfab" />

### 🔹 PuTTY 연결 준비
- Jetson 보드에 SD카드를 삽입하고 전원을 연결한 후, C타입 케이블로 Host PC와 연결  
- Host PC의 장치 관리자에서 **포트(COM & LPT)** 항목을 열어 직렬 장치 포트 번호 확인  

### 🔹 PuTTY 설정
- PuTTY 실행 후 **Serial** 타입으로 연결 설정  
- 확인한 포트 번호(COMx)를 입력  
- **Speed(전송 속도)** 는 `115200`으로 설정  

### 🔹 연결
- 설정 완료 후 **Open 버튼**을 클릭하여 Jetson Orin Nano와 연결 진행  

<img width="1034" height="304" alt="image" src="https://github.com/user-attachments/assets/d1ed9290-a3b1-4222-bd98-eb723dc990e5" />

### 🔹 PuTTY 초기 설정
- Jetson 보드 부팅 후 PuTTY 창에서 **NVIDIA Software License** 동의 화면 출력  
- 키보드의 방향키와 Spacebar, Enter 키를 사용하여 선택 및 진행  

### 🔹 라이선스 동의 절차
- "License For Customer Use of NVIDIA Software" 내용을 확인  
- 동의 후 `OK` 버튼을 선택하여 초기 설정 단계로 진입  

<img width="790" height="462" alt="image" src="https://github.com/user-attachments/assets/b079eb59-d164-409b-b821-7074df98742c" />

### 🔹 언어 설정
- 설치 과정에서 사용할 기본 언어를 선택  
- **English** 로 설정  

### 🔹 지역 설정
- Location 선택 단계에서 **other** 선택  
- Continent or region: **Asia** 선택  
- Country: **Korea, Republic of** 로 설정  

<img width="1037" height="308" alt="image" src="https://github.com/user-attachments/assets/5ae6508d-8b2f-48ca-9476-9f6f09b0ece2" />

### 🔹 Locales 설정
- 설치 언어 및 국가에 맞는 **기본 로케일(Locale)** 선택  
- 예: `United States - en_US.UTF-8` 등 적절한 로케일 지정  

### 🔹 System Clock 설정
- 시스템 시간이 **UTC(Coordinated Universal Time)** 기준으로 설정됨  
- 권장 설정: `Yes` 선택하여 UTC 기반 시간 동기화 진행

<img width="792" height="464" alt="image" src="https://github.com/user-attachments/assets/325516c7-f936-4690-b8bc-1f0c836754c7" />

### 🔹 사용자 계정 생성
- 새 사용자 계정의 **이름(Full name)** 입력  
- 계정에 사용할 **Username** 지정  

### 🔹 비밀번호 설정
- 사용자 계정의 비밀번호는 1212로 고정
- 동일한 비밀번호(1212)를 다시 입력하여 확인

<img width="516" height="298" alt="image" src="https://github.com/user-attachments/assets/fa29dbe5-9564-4bf3-810e-fe1d20df3e51" />

### 🔹 APP 파티션 개요
- APP 파티션은 운영체제 및 애플리케이션 파일이 저장되는 영역  
- Jetson 보드 초기 설정 시 입력 값에 따라 파티션 크기를 지정 가능  

### 🔹 설정 방법
- 해당 값을 **빈칸으로 두면** SD카드의 사용 가능한 용량 전체를 APP 파티션으로 할당  
- 권장: 빈칸 입력 → SD카드 용량 전체를 APP 파티션으로 설정

<img width="791" height="460" alt="image" src="https://github.com/user-attachments/assets/ecc3eaa5-b481-4f06-896b-f06bb5a8fb71" />

### 🔹 네트워크 설정
- Jetson Orin Nano는 기본적으로 **Wi-Fi 모듈**이 내장되어 있음  
- 초기 셋업에서는 **dummy0 인터페이스**를 선택하여 Wi-Fi 연결을 진행하지 않도록 설정  

### 🔹 주의 사항
- DHCP 자동 설정 과정에서 실패 메시지가 나타날 수 있음  
- 이 경우, **네트워크를 설정하지 않음(Do not configure the network at this time)** 옵션을 선택하여 설치를 계속 진행  

<img width="790" height="459" alt="image" src="https://github.com/user-attachments/assets/94d39d8f-efe9-4d1e-ad65-20fbbf35e9ff" />

### 🔹 시스템 이름 설정
- 보드의 **Hostname**을 `ubuntu` 로 지정  
- 네트워크 상에서 장치를 구분하는 식별자로 활용됨  

### 🔹 Chromium 브라우저 설치
- 초기 설정 과정에서 **Chromium Browser** 설치 여부 확인  
- `Yes` 선택 시 설치 진행 (시간이 다소 소요됨)

<img width="1029" height="297" alt="image" src="https://github.com/user-attachments/assets/b090d304-c287-434c-a997-677d9910e88e" />

### 🔹 설치 진행 시 주의사항
- 네트워크 미연결로 인해 발생하는 **Error 메시지**는 무시하고 설치를 계속 진행  
- 예: `ERROR: Cannot connect to snap store.` 발생 시 `OK` 선택  

### 🔹 연결 상태
- 설치 과정 중 일시적으로 **보드와의 연결이 해제**될 수 있음  
- 정상적인 동작이므로 설치 완료까지 기다리면 됨  

<img width="1028" height="300" alt="image" src="https://github.com/user-attachments/assets/f191ebc6-bf17-467d-823e-8a797a2e24ac" />

### 🔹 보드 재연결
- 장치 관리자에서 보드와의 재연결이 확인되면 다시 **PuTTY**를 실행하여 통신  
- 셋업 과정 중 설정한 **사용자 계정(username)** 과 **비밀번호(password)** 로 로그인  

### 🔹 로그인 주의사항
- Linux 환경에서는 비밀번호 입력 시 화면에 표시되지 않음  
- 입력 후 **Enter** 키를 눌러 접속 진행  

<img width="1029" height="301" alt="image" src="https://github.com/user-attachments/assets/162f1c9a-58a1-47ea-8290-2a09f7355a3c" />

### 🔹 nmcli 개요
- `nmcli`는 Linux에서 네트워크 관리자를 제어하는 도구  
- GUI 없이도 네트워크 설정 및 관리 가능  

### 🔹 Wi-Fi 네트워크 검색
- 명령어:  
  ```bash
  nmcli d wifi list
  ```
- 현재 연결 가능한 Wi-Fi 네트워크 리스트 출력

### 🔹 명령어 종료
- Ctrl + C 입력 시 Wi-Fi 네트워크 검색 리스트 종료 가능

<img width="1029" height="298" alt="image" src="https://github.com/user-attachments/assets/09e3c82a-50d2-46fe-9b4b-2cf3076ab911" />

### 🔹 Wi-Fi 연결
- `$ nmcli d wifi connect [Wi-Fi 이름] password [비밀번호]` 명령어를 사용하여 Wi-Fi에 연결  
- 연결 성공 시, Jetson 보드는 부팅할 때 해당 Wi-Fi에 자동 접속  

### 🔹 인터넷 연결 확인
- `$ ping 8.8.8.8` 명령어를 사용하여 Google Public DNS 서버로 인터넷 연결 상태 확인  
- 응답이 정상적으로 돌아오면 네트워크 연결이 정상 동작  

### 🔹 ping 테스트 종료
- **Ctrl + C** 입력 시 `ping` 프로세스를 종료할 수 있음

<img width="1028" height="299" alt="image" src="https://github.com/user-attachments/assets/17b757a1-93d8-4935-9896-4432851ce4dd" />

### 🔹 네트워크 인터페이스 확인
- `$ ifconfig` 명령어를 통해 네트워크 인터페이스 정보를 확인  
- `ifconfig`는 Linux에서 네트워크 인터페이스를 설정하고 관리하는 명령어  

### 🔹 Wi-Fi 연결 인터페이스
- Wi-Fi를 통해 연결된 네트워크 인터페이스는 `wlan0`  
- `inet` 항목에서 할당된 **IP 주소**를 확인 가능  

### 🔹 작업 종료
- 확인을 마친 후 **PuTTY**를 종료  

---
# SSH 연결
---

<img width="1067" height="302" alt="image" src="https://github.com/user-attachments/assets/83110aa0-4c87-4c97-a027-f4d0aeaa55d2" />


### 🔹 SSH 접속 방법
- Windows의 **PowerShell**을 이용하여 SSH 연결 진행  
- `ssh [사용자계정]@[보드IP주소]` 입력 후 보드 접속  

### 🔹 접속 시 주의사항
- 장치 fingerprint 등록 시 **yes** 입력  
- 이후 비밀번호 입력 후 접속 진행  

---
# Linux
---

<img width="533" height="297" alt="image" src="https://github.com/user-attachments/assets/a0edd7cd-21d6-472e-b2f0-3bde920a61e8" />

### 🔹 기본 개요
- Jetson Orin Nano는 Ubuntu Linux 기반의 **Jetpack OS** 사용  
- Linux에서는 모든 설정과 파일이 **디렉토리**로 구성됨  
- 터미널 환경에서는 항상 `현재 위치 디렉토리` 개념이 존재  

### 🔹 홈 디렉토리
- `~` 는 Home 디렉토리의 현재 위치를 의미  
- Home 디렉토리 내부 파일과 폴더를 기준으로 작업 수행  

### 🔹 파일 및 디렉토리 확인
- `$ ls` : 현재 위치의 파일 및 하위 디렉토리 리스트 출력  
- `$ ls -a` : 숨김 파일 포함 출력  
- `$ ls -l` : 파일 권한, 소유자, 크기, 수정 시간 등 상세 정보 출력  
- `$ ls -la` : 숨김 파일 + 상세 정보 모두 출력  

### 🔹 확인 사항
- Linux 명령어는 다양한 옵션과 함께 사용 가능  
- 동일한 명령어라도 옵션 조합에 따라 결과가 달라짐  

<img width="807" height="288" alt="image" src="https://github.com/user-attachments/assets/d768ad8d-b60d-492b-8e69-32532e04e047" />

### 🔹 Root 디렉토리
- 모든 디렉토리는 **Root 디렉토리(`/`)** 아래에 존재  
- 계층적인 구조(Tree Structure)를 가짐  

### 🔹 주요 디렉토리
- **bin** : 기본 실행 파일들이 위치하는 디렉토리  
- **usr** : 사용자 관련 프로그램과 라이브러리 저장  
- **sys** : 시스템 관련 파일들이 위치  

### 🔹 Current Directory
- 터미널에서 작업할 때 항상 **현재 디렉토리(Current Directory)** 개념이 존재  
- 현재 디렉토리는 작업의 기준점이 되며, 빨간 박스로 표시된 위치에 해당 

---

<img width="1064" height="298" alt="image" src="https://github.com/user-attachments/assets/58ae7c58-8780-49ef-af4f-311279244177" />

### 🔹 cd 명령어
- `cd` 명령어는 현재 위치(작업 디렉토리)를 변경하는 명령어  
- `cd ..` 입력 시 상위 디렉토리로 이동 가능  

1. `cd ..` 명령어를 사용하여 Root 디렉토리(`/`)로 이동  
2. Root 디렉토리에서 하위 디렉토리들을 확인 (`ls -al`)  
3. 다시 `cd home` → `cd lab`을 입력하여 `home/lab` 디렉토리로 이동  
- Root 디렉토리(`/`)에는 어떤 하위 디렉토리들이 포함되는지 조사하라

---

<img width="534" height="299" alt="image" src="https://github.com/user-attachments/assets/1d1f78c3-f72f-477e-81fa-34f0f0f921f5" />

### 🔹 mkdir 명령어
- `mkdir` 명령어는 새로운 디렉토리를 생성할 때 사용  

```bash
# mkdir test1       # home 디렉토리 아래에 test1 디렉토리 생성
# ls -al            # 디렉토리 생성 여부 확인
# cd test1          # 생성한 디렉토리 내부로 이동
```

---

<img width="1066" height="301" alt="image" src="https://github.com/user-attachments/assets/20fedde9-5990-4d2e-82f9-8a8671d5d8ac" />

### 🔹 Vim install
- Apt는 Ubuntu에서 패키지 관리를 위한 툴이다. 이를 이용하여 소프트웨어 패키지를 관리할 수 있다.  
- `$sudo apt update` 명령어를 사용하여 설치 가능 패키지 목록을 업데이트 한다.  
- 이후 `$sudo apt install vim` 명령어를 사용하여 vim을 설치한다.  
- Vim은 텍스트 편집기로 터미널 환경에서 텍스트 파일 편집 시 사용한다.  

1. `$sudo apt update` 명령어를 입력하여 패키지 목록 업데이트  
2. `$sudo apt install vim` 명령어를 입력하여 Vim 설치 진행  
3. 설치 완료 후 `vim --version`으로 정상 설치 확인  

<img width="1067" height="300" alt="image" src="https://github.com/user-attachments/assets/2262d1f1-00de-4180-9b96-a4380e5aa9f6" />

### 🔹 Vim 사용
- `vim [파일명].txt` 명령어로 새로운 텍스트 파일 생성 가능  
- 실행 후 `i` 키 입력 시 **편집 모드**로 전환  
- 입력 완료 후 `[Esc]` 키를 눌러 **명령 모드**로 전환  
- `:wq` 입력 시 파일 저장 후 종료  

1. `$ vim test1.txt` 명령어 입력하여 새 파일 생성  
2. `i` 키 입력 후 이름과 학번 입력  
3. 입력 완료 후 `[Esc]` → `:wq` 입력하여 저장 및 종료  
4. `cat test1.txt` 명령어로 저장된 내용 확인  

---

<img width="533" height="299" alt="image" src="https://github.com/user-attachments/assets/0300ab7c-d45c-4a70-8abe-a04f9b755477" />

### 🔹 Jetson-stats install
- **Jetson-stats**는 Jetson 계열 보드에서 시스템 상태를 모니터링하고 제어할 수 있는 패키지  
- `sudo apt install python3-pip` 명령어를 사용하여 pip(파이썬 패키지 관리자) 설치  
- `sudo pip3 install -U jetson-stats` 명령어를 사용하여 최신 버전 jetson-stats 설치  
- `-U` 옵션은 최신 버전으로 업데이트하여 설치함  

1. `$ sudo apt install python3-pip` 명령어 입력 → pip 설치  
2. `$ sudo pip3 install -U jetson-stats` 명령어 입력 → jetson-stats 최신 버전 설치  
3. 설치 완료 후 `$ jtop` 명령어 실행 → Jetson 보드 상태 모니터링 확인  

### 🔹 재부팅
- `sudo reboot now` 명령어를 사용하여 보드를 즉시 재부팅 가능  
- `shutdown` 명령어를 사용하면 보드를 종료할 수 있으며, `reboot` 명령어로도 재부팅 가능  

1. `$ sudo reboot now` 입력 → Jetson 보드 재부팅  
2. `$ sudo shutdown now` 입력 → 보드 종료  
3. `$ sudo reboot` 입력 → 보드 재부팅

<img width="532" height="300" alt="image" src="https://github.com/user-attachments/assets/c980ab92-2a10-4307-863a-d25a38740f2e" />

### 🔹 Jtop 실행 및 모니터링
- `jtop` 명령어 실행 시 Jetson 보드의 실시간 상태 모니터링 가능  
- CPU, GPU, 메모리 사용량 및 온도, 전력 소비 등을 직관적으로 확인 가능  
- Jetson 개발 환경에서 성능 및 자원 상태를 관리하는 데 유용  

1. `$ jtop` 입력 후 실행  
2. CPU/GPU 사용량, 메모리 상태, 전력, 센서 온도 확인  
3. 종료 시 `q` 키 입력
