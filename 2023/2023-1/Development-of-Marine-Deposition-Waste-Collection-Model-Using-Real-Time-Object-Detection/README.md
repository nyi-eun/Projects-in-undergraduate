<h2 align='center'> 실시간 객체 탐지를 활용한 해양침적폐기물 수거 모델 개발 </h2>
<h3 align='center'> [전공] 딥러닝 Project </h3>
<h4 align='center'> (2023.05. ~ 2023.06.) </h4>  

![Aqua Lines](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

&nbsp;

## 1. 배경 및 목적

- 해저에 침적되어 있는 폐기물을 실시간으로 탐지할 수 있는 딥러닝 모델 개발 (아이디어 프로젝트)  
- RTDeSNet(모델 개발명) → 수중 드론에 탑재 → 침적쓰레기 수거 효율 증대 기대
  
<br/>

## 2. 주최 기관

- 주최: AI빅데이터융합경영학과 전공 수업  ‘딥러닝’

<br/>

## 3. 프로젝트 기간

- 2023.05 ~ 2023.06 (2개월)


<br/>

## 4. 프로젝트 설명 
![](https://github.com/Ji-eun-Kim/Text-Data-Analytics/assets/124686375/b9da5660-e83d-4290-a8f0-9cb56fe5362f)

해저에 침적되어 있는 폐기물을 실시간으로 탐지할 수 있는 딥러닝 모델을 개발하여, 침적 쓰레기를 수거하는데 보다 효율성을 증대시키기 위한 방안으로 ‘실시간 객체 탐지를 활용한 해양침적폐기물 수거 모델 개발’을 제안하였다. 이를 통해 침적 폐기물 수거의 증가 및 수거 인력 예산안 문제, 폐그물로프와 폐어구로 인한 선박사고 발생 문제 우려에 대한 문제점들을 해결할 수 있을 것이라 기대된다.

   기존 연구의 한계점으로는 1. YOLO 모델에 국한되어 Object detection을 수행한다는 점, 2. 로프 등의 polygon 형식 데이터에 대해 Segmentation이 불가하다는 점, 3. Data의 전처리 부족, 4. Detection과 Segmentation을 동시에 수행할 수 없다는 점, 5. Bbox와 Polygon 데이터 각각에 다른 모델을 적용할 경우, Inference 속도가 저하된다는 점을 고려하여 기존의 연구로부터의 한계점을 보완하기 위해 다음과 같은 방안을 제시하였다.

   해양침적쓰레기 9종에 대한 111,890장의 이미지 데이터인 **AI Hub의 <해양침적쓰레기 이미지 데이터 고도화> 데이터 셋을 학습**에 사용하였다. 대부분의 어망과 루프는 흙 속에 묻혀 있어 Detection만으로는 찾기가 어렵다는 단점을 보완하기 위해 어망과 루프는 Segmentation을 활용하고자 하였는데, 이 데이터셋은 어망류와 로프류가 Segmentation의 형태로 라벨링이 되어 있어 본 태스크의 목적에 적합하다고 판단하였다. 보다 효과적인 학습을 위해 전처리는 2가지를 진행하였다. 부유물로 인한 화질 저하를 개선시키기 위한 **Denoising**으로 모델은 **U-Net**을 활용, 지나치게 어두운 이미지에 대한 밝기를 조절해주기 위한 **Low-Light Image Enhancement**으로 모델은 **DCE-Net**을 활용하였다.

   모델링의 경우, Detection과 Segmentation을 동시에 수행할 수 있는 모델 구조를 제안하였다. 우선, Bbox를 사용하지 않고 바로 Segmentation을 수행하며, 다양한 크기의 feature map을 활용하는 기존 **RTMDet** 모델의 **Head 부분**을  **Condlnst**로 대체하였다. **다양한 크기의 feature map을 활용할 수 있다는** 점을 고려 Condlnst로 선정하였으며, **Bbox를 포함해 4개의 head로 구성**되어 보다 **효과적인 정보를 습득할 수 있다는 장점**을 지닌다. 기존 Condlnst Loss의 경우, Detection 클래스와 Segmentation 클래스를 구분하지 못한다는 점을 고려해 **클래스 label 값(루프 및 어망의 경우, Segmentation으로 라벨링)을** 통해 **Controller Head의 사용 여부를 결정**하였다. 결론적으로 Mask를 생성할 클래스와 생성하지 않을 클래스를 구분하여 Detection만 수행할 것인지 Segmentation을 수행할 것인지를 선택적으로 수행할 수 있도록 모델 구조를 변경하였다.  

   결론적으로 **데이터 전처리**는 **Denoising, Low-Light Image Enhancement,** **모델링은** **RTMDet(CSP-Net+PAFPN) + Condlnst**을 사용하였다. 이를 통해 어망과 루프를 제외한 해양 쓰레기의 경우 Detection 기법을 사용하여 쓰레기를 수거하도록 하였고, 보다 찾기 어려운 어망과 루프는 Segmentation을 수행하여 잘 탐지할 수 있도록 하였다. 


<br/>


### <구현 예시 시각화>
![un](https://github.com/Ji-eun-Kim/Text-Data-Analytics/assets/124686375/8ab27274-d072-46ae-87e5-fef3c2d188c8)
  
  
## 5. 팀원 및 담당 역할  

<팀원>  
- 전공생 4명

  <br>
  
<담당 역할>
- 전반적인 프로젝트 흐름 설계
- 기존 연구 한계점 제시
- Data preprocessing(Denoising/Low-Light Image Enhancement) 모델링 아이디어 및 기법 제시
- Segmentation & Detection(RTMDet/Condlnst)의 동시 수행 모델링 아이디어 및 기법 제시
- PPT 제작

<br/>

## 6. 발표 자료

- 발표 자료    
https://drive.google.com/file/d/1tqsU1VrPUPqiPklEI7EA7L5sDnrjC3bN/view?usp=drive_link

- 추가) 선행연구조사 발표 자료-YOLO  
https://drive.google.com/file/d/1c-IkR-SlKpHeUMiiNud2FhfldH3eNme8/view?usp=drive_link
