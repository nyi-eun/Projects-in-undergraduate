<h2 align='center'> Object Detection을 활용한 자동차 운전자 이상행동탐지 
<h3 align="center"> [전공] 비전AI와비지니스 Project </h3>  
<h4 align='center'> (2023.11. ~ 2023.12.) </h4>  

![Aqua Lines](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)  

&nbsp;

## 1. 배경 및 목적

- 전체 사망자 비율 중 70% 이상이 졸음 및 주시 태만으로 인해 발생
- 졸음운전 및 주시 태만(휴대전화 사용)을 감지하는 시스템의 필요성
- 안전 운행을 위한 자동차 운전자 이상행동 감지 시스템화

<br/>

## 2. 주최 기관   

- 주최: AI빅데이터융합경영학과 전공 수업 ‘비전AI와 비지니스’  

<br/>

## 3. 프로젝트 기간   

- 2023.11 ~ 2023.12 (2개월)

<br/>

## 4. 프로젝트 설명    
![화면 캡처 2023-12-30 223407](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/5e8b15f5-206b-4cee-a61c-dd9725bf9751)  

Object Detection을 활용한 자동차 운전자 이상행동탐지 딥러닝 프로젝트를 진행하였다. 이를 통해 운전자의 **비정상적인 행동을 감지**하고 **즉각적인 조치**를 취함으로써 도로 상황에서의 **잠재적인 사고를 예방**할 수 있을 것이라 기대된다.    

연구의 Flow는 다음과 같다. Image Sequence를 input으로 하여 **SOTA** 를 달성한 **YOLOv8** 를 통해 감은 눈 & 소지품, 신체를 Detection하고, Detection된 정보들을 토대로 휴대폰 사용 여부와 **운전자의 졸음 지수 기존 초과 여부를 통해 이상행동을 정의**하였다. 이에 따라, **휴대폰 감지 여부와 졸음 지수**는 프로젝트의 핵심 측정 지표로 활용되며, 이상 행동을 정의하기 위한 기준으로 사용된다. 설정된 기준을 초과하는 경우, 휴대폰 사용이나 졸음이 뚜렷하게 감지되는 상황에서는 시스템이 해당 행동을 이상으로 간주하여 운전자의 안전성 향상 및 교통사고 예방에 기여하는데 중점을 두었다.    

해당 프로젝트에 학습으로 사용된 데이터셋은 총 **2개의 데이터셋**으로, AI HUB의 **1. 졸음운전 예방을 위한 운전자 상태 정보 영상 데이터셋 - 66,358장**과 **2. 운전자 및 탑승자 상태 및 이상행동 모니터링 데이터셋 - 237,844장**이 사용되었다.  

우선 각 데이터의 형태는 Image Sequence이었고, 본 프로젝트의 목적에 맞게 각 데이터에 대해서 **전처리**를 수행하였다. **졸음운전 예방을 위한 운전자 상태 정보 영상 데이터셋**의 경우, x,y 좌표가 이미지 범위를 초과하는 경우가 일부 존재하여, **w,h 이미지 범위 내로 조정 후 scaling**을 진행하였다. **운전자 및 탑승자 상태 및 이상행동 모니터링 데이터셋**의 경우, 필요한 정보만 추출하여 학습에 사용하였고, 기존 비디오 정보 중, **졸음 운전, 휴대폰 조작을 이상행동**으로 정의하였고, 이상행동과 정상행동으로 분류 및 라벨링하여 이상행동탐지에 활용하였다.  

![화면 캡처 2023-12-30 215149](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/0fa4128e-3f12-40a5-b0ef-493cdc49bb33)  
모델 학습의 경우, Detection task에서 **SOTA를 달성한 YOLOv8**을 채택하였고, **Train:Test = 8:2** 로 나누어 학습을 진행하였다. 진행한 결과는 5.에서 확인할 수 있다. 비교적 Detection을 하기 쉬운 datasets으로 구성되어 있어, **epoch을 10**로만 해도 성능이 월등히 잘 나올 수 있었던 것 같다. Inference의 경우, 한 이미지당 **30초 내외**로 결과가 나오는 것 또한 확인할 수 있었다.  

![55](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/c1936b40-cb0d-4f56-87fe-59937d2ed116)|![66](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/700cc1d5-d2e0-44d1-8f1c-c78d614027ac)
---|---|

다음은 이상행동 탐지에 대한 Process이다. 이상 행동은 휴대폰 조작과 졸음 운전 두 가지로 정의하였다. 휴대폰 조작의 경우, 감은 눈 & 소지품 Detection model을 통해 **시퀀스 내 threshold 이상의 확률로 휴대폰이 감지되면 즉시 이상행동으로 처리**하도록 구현하였다. 졸음운전의 경우, 감은 눈 & 소지품 Detection model을 통해 **(오른쪽 감은 눈 감지 확률 + 왼쪽 감은 눈 감지 확률)/2**로, 운전자 신체 Detection model을 통해서는 **(몸 Bbox의 a 지점 아래의 얼굴 Bbox 길이)/(얼굴 Bbox 전체 길이)**로 정의하여, **두 결과의 평균이 threshold 이상인 이미지가 해당 시퀀스 내 일정 비율 이상인 경우** **해당 시퀀스를 이상행동으로** 처리하도록 구현하였다.  

따라서, 본 모델링에 설정한 파라미터는 **1. 휴대폰 감지 확률 threshold, 2. 몸 Bbox 기준 지점, 3. 두 결과 평균에 대한 threshold, 4. 시퀀스 내 졸음판정 비율**이 사용되었으며, **평가지표**의 경우, 기존 연구들과 동일한 지표인 **Accuracy**와 이상행동 탐지 성능 확인을 위한 **Recall**을 함께 확인하였다. 실험한 결과, **0.85, 0.2, 0.12, 0.15**의 파라미터 값으로 설정하였을 때, **Accuracy는 0.6625, Recall은 0.6875로** 두 가지 측면에서 가장 좋은 결과가 도출됨을 확인하며 해당 프로젝트를 마무리하였다.  

<br/>

<!-- ## 5. 결과  
![Untitled](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/c61dd550-80b9-4991-9cea-966e43ff5317)|![222](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/35efccc5-d388-4b15-bc58-873487d4c992)
---|---|

![3333](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/72dde112-4e84-4dc5-b0fc-4ff14821cbde) | ![444](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/1808cd96-b3db-4ca9-a36e-df983c6b4d19)
---|---|

![66](https://github.com/Ji-eun-Kim/VisionAI_project/assets/124686375/531eef24-33ad-4c22-a87b-2f2a8f96cfc2)-->


## 6. Contribution 및 발전 방향    

<시사점>   
- 졸음 운전 판단 시 몸/얼굴 Bbox (두 가지 정보) 활용    
- Detection 모델을 활용한 운전자 모니터링 시스템 방안 제안  

<발전 방향>     
- 영상 데이터 2종류 모두 확보하지 못하여 영상 → 이미지 시퀀스 → 이미지 단위 처리 진행  
    - 영상 단위 처리를 통한 실시간성 확보  
- 한정된 데이터 내에서만 이상행동 및 판정 기준 설정 진행  
    - 연구를 통해 더 다양한 이상행동에 대한 데이터 확보 및 위험운전 기준 설정  

## 7. 팀원 및 담당 역할    

<팀원>   
- 전공생 3명    

<담당 역할>    
- 운전자 이상행동 데이터 셋 전처리
- YOLOv8 모델 학습 및 최적화 
- 운전자 이상행동 기준 아이디어 제시
- 전반적인 흐름 설계
  
<br/>


## 8. 발표 자료

https://drive.google.com/file/d/1KdXlEvLr6ixGDgle1ipbMQzmSQOHLBeK/view?usp=drive_link
