<h2 align='center'> 이미지 내 문구 번역 및 원본 스타일 적용 </h2>
<h3 align='center'> [인공지능학회 X:AI] Toy Project  </h3>  
<h4 align='center'> (2023.07. ~ 2023.08.) </h4> 

![Aqua Lines](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)


&nbsp;

## 1. 배경 및 목적

- 이미지 내 문구 번역 및 원본 스타일 적용 (Toy Project)
- (영어 → 한국어) 상호 이해와 소통을 원활하게 도와주는 효과 기대

<br/>

## 2. 주최 기관

주최/주관: AI빅데이터융합경영학과 인공지능 학회 X:AI 

<br/>

## 3. 프로젝트 기간 
- 2023.07 ~ 2023.08 (2개월)


<br/>

## 4. 프로젝트 설명 
<img width="1022" alt="스크린샷 2023-09-01 155417" src="https://github.com/Ji-eun-Kim/Toy_project/assets/124686375/683aafb4-b8e2-45dd-9c63-b771c6acb027">

영어로 된 간판 및 표지판 이미지의 텍스트를 영→한 번역하여 배경만 남은 이미지와 합성하는 딥러닝 프로젝트를 진행하였다. 이를 통해 상호 이해와 소통을 원활하게 도와주는 효과 기대할 수 있을 것이라 기대된다.

Task는 총 3가지로, 1. **[DBNet, DBNet++, PSENet, FCENet]**(Detection), **[SAR, ABINet, SATRN]**(Recognition) 2. **[T5]**(Text translation) 3. **[SynthText]**(Image Composition) 로 진행하였다. 진행 방법은 다음과 같다.

1. Detection & Recognition model을 활용해 해당 텍스트에 대해 Bbox 좌표값과 해당 영어 text를 output 값으로 뽑아낸다.
2. Detection&Recognition로부터 나온 영어 text를 한글로 번역해주도록 NLP model인 **T5**를 학습 시킨 후, 번역된 한글 text를 output으로 뽑아낸다. 
3. Detection model의 Bbox를 기반으로 텍스트를 지워주는 model인 **Stroke-based Scene Text Erasing**로 배경의 텍스트를 지운다. 
4. Detection의 bbox 좌표값과, T5의 한글 text, Stroke-based Scene Text Erasing의 지워진 배경을 input으로 받아, **OpenCV 라이브러리를 활용**해 SynthText를 진행한다.

&nbsp;

1. **Detection & Recognition**

텍스트의 bbox 좌표값을 알아내기 위해 Detection model을 사용하였다. Detection model 중에서도 text가 기울어져 있거나 회전이 되어있는 text에 강건한 특징을 지니고 있는 model들을 추가적으로 선정하여 **DBNet, DBNet++, PSENet, FCENet**을 사용하였다. Recognition model의 경우에도 text의 curve, orientation에 구애 받지 않고 인식이 가능한 **SAR**과, 언어적 사전 지식을 바탕으로 text를 인식하는 **ABINet,** 문맥을 고려하여 Recognition을 수행하는 **SATRN**을 활용해 Text를 추출해주었다. 

Dataset의 경우, 영어 간판 및 표지판 데이터 셋인 **TextOCR dataset**과, **Incidental Scene Text dataset**을 활용해 학습에 사용하였다.  1차 output으로는 Detection의 bbox 좌표값과 Recognition을 통한 Text를 txt 파일로 추출해주는 코드를 구현하여 output을 뽑아냈다. 

2. **Text translation**

영어를 한글로 번역하기 위해 NLP 모델인 **T5**를 사용하였다. 기존의 경우, **Transformer**을 활용해 번역 text를 뽑아내고자 하였으나, **영어 토크나이저 문제 및 환경 설정 충돌**로 인해 모든 텍스트 기반 언어 문제를 텍스트 대 텍스트 형식으로 변환하는 T5를 사용하였다. 

T5-base 모델은 **메모리 에러로** 인해 fine-tuning을 통한 성능 고도화의 어려움이 존재하였고, 따라서 T5-base에 한국어 fine-tuning이 된 **AIRC-KETI/Ke-T5**을 사용하여 output을 뽑아냈다. 이 때 **T5 모델은 주로 문장 단위의 번역**을 수행하는 것과 달리 **Recognition model이 요구하는 input 형식은 단어 형태**이었기에, T5 input을 문장 형태로 만들어 주기 위해 단어 끝에 마침표를 넣어 문장처럼 보이도록 input을 변경해주었다. output은 띄어쓰기로 단어를 구분하여 순서대로 매칭된 값을 output으로 넘겨주도록 하였다.  

3. **Image Composition**

Image Composition의 경우, 두 분야로 나누어 진행하였다. **1. 텍스트 지워주는 model, 2. 번역된 text와 배경을 합성하는 model.** 기존에는 원본 이미지 내 텍스트의 색상 , 서체 등의 스타일과 배경 이미지를 그대로 유지한 채 합성해주는 **SRNet**을 사용하려고 하였으나, **한글 데이터의 fine-tuning이 되지 않는 문제가 발생**하여 대체 모델을 탐색하였다. model은 총 2가지를 사용하였고, 원본 이미지의 텍스트를 **Stroke-based Scene Text Erasing**을 통해 지워주었으며, **SynthText(OpenCV) 라이브러리**를 통해 원본 텍스트의 스타일을 유지하며 배경 이미지 위에 텍스트를 합성하였다. 

이때, 영어 텍스트의 스타일을 한글 텍스트에 적용하기 어렵다는 한계가 존재해 텍스트 서체 유지 대신 원본 text의 색상 및 크기 유지를 하는 방향으로 진행하였다.

<br/>

## 5. 결과
<img src="https://github.com/Ji-eun-Kim/X-AI_Toy_project/assets/124686375/f4a56b3b-1c62-42fe-af48-b9c8efa7b1fd"> | <img src="https://github.com/Ji-eun-Kim/X-AI_Toy_project/assets/124686375/a148bc40-393a-4c36-92c9-b21aa184368d">
---|---|

<img width="611" alt="KakaoTalk_20230901_002124391_10" src="https://github.com/Ji-eun-Kim/Text-Data-Analysis/assets/124686375/58c0a4bd-89cb-4437-8f89-cddec9b61569"> | <img width="611" alt="KakaoTalk_20230901_002124391_11" src="https://github.com/Ji-eun-Kim/Text-Data-Analysis/assets/124686375/20f7c602-7c6a-48d8-8335-9b2c95b9b3f8">
---|---|

&nbsp;

## 6. 느낀점 및 한계점

8주라는 짧은 기간 동안 진행한 프로젝트였기 때문에 결과물에 대한 만족감이 부족했다. 그럼에도 제한된 환경 속에서 노력하고 협력하여 새로운 경험을 얻고 배우며 성장할 수 있게 되었다. 

각기 **다른 모델 환경 설정 문제**로 인해 **End to End 학습이 어려워** 개별 모델의 성능에 따라 최종 결과물의 품질이 크게 달라진다는 어려움이 존재했다. 따라서, 추후 end-to-end 형식의 py 파일로 개선하고, 한 번에 효율적으로 결과물을 얻을 수 있도록 계획하고 있다. 

이러한 어려움을 극복하면서 훌륭한 협력과 문제 해결 능력을 발휘하였으며, 앞으로의 프로젝트에 대한 자신감을 가지고 나아갈 수 있을 것이라 기대한다.


&nbsp;

## 7. 팀원 및 담당 역할  

<팀원>  
- 전공생 6명   

<br>
  
<담당 역할>    
- Image & Text Composition 
- SRNet 인퍼런스 및 한글 데이터 Fine tuning
- SRNet 대체 방안 모색
- 이미지 + 텍스트 합성 코드 구현

<br/>

## 8. 발표 자료 및 깃허브

- 발표 자료  
https://drive.google.com/file/d/1rOM6w_Og76kaWW6GRzes2DgshZWrrDUQ/view?usp=sharing

- 깃허브    
https://github.com/Ji-eun-Kim/X-AI_Toy_project/blob/main/README.md
