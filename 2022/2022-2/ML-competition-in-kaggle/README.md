<h2 align='center'> 인사 정보 데이터 기반 연봉 예측 모델링 </h2>
<h3 align='center'> [전공] 머신러닝 Competition </h3>
<h4 align='center'> (2022.10. ~ 2022.11.)  

![Aqua Lines](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png)

&nbsp;


## 1. 배경 및 목적

- 인사 정보 데이터로 연봉 예측하는 회귀 문제
- 평가지표: NMSE

<br/>

## 2. 주최 기관

- 주최: AI빅데이터융합경영학과 전공 수업 ‘머신러닝’
- 주관: Kaggle Inclass Competition
- 순위: 6위 (16팀 중)
- 참가 인원: 48명(16팀)

<br/>

## 3. 프로젝트 기간

- 2022.10 ~ 2022.11 (2개월)
  
<br/>

## 4. 프로젝트 설명 
인사 정보에 대한 데이터 셋으로 각 **연봉을 예측하는 회귀 문제**이다. 결측치의 경우 4개의 column에서 나타났다. **직무태그**의 경우, 공란 칸을 지우지 않고, ‘없음’으로 채웠으며, **근무형태**의 경우 분석을 통해 ‘계약직’으로 처리하였다. **어학시험**의 경우, 빈칸을 ‘미응시’로 처리하였고, **대학성적**의 경우 describe()를 찍어서 확인해본 결과 **중앙값 대체**가 적절하다고 판단하여 결측값들을 중앙값으로 전처리를 진행했다. 피처 엔지니어링의 경우, **수치형 피처 203개, 범주형 피처 38개**로 총 **241개의 피처**를 생성하였다. PCA나 Feature Selection을 추가적으로 시도했으나, 시도하지 않았을 때의 성능이 가장 좋아, 적용하지 않은 결과를 활용하였다.  

   모델링의 경우, **선형 회귀**(LinearRegression, Lidge, Lasso, ElasticNet), **트리 계열**(DecisionTree, RandomForest, XGBoost, LGBM, Catboost, ExtraTree)을 사용하여 실험을 진행했다. 각 모델마다 알맞은 **Pipeline**을 직접 생성하여 학습을 진행해주었다. 결과적으로는 **CatBoost, LGBM, ExtraTree** 모델로 학습했을 때 성능이 가장 좋아 3가지 모델을 최종 모델로 선택하였다. 

   우선 **Catboost**의 경우, category 피처를 자동으로 처리해준다는 특징을 생각했을 때, 우리 데이터에 적합한 모델이라 판단하여 이 모델을 선택하였다. 모델링 후, **KMeans**로 군집화하여 시도해보기도 하였다. **LGBM**의 경우, Catboost와 비슷한 **Boosting 계열**의 모델이라는 생각에 착안해 Catboost 단일 최고 성능이 나온 피처들에 피처를 조금 더 추가하였고, 전처리도 약간 변경하여 모델링을 진행하였다. 이는 앙상블 시에 모델 간 **상관관계**를 줄이기 위한 아이디어로 진행하게 되었다. 마지막으로 ExtraTree의 경우, 위의 부스팅 계열과는 조금 다른 유형의 모델을 사용하면 보다 성능 향상에 도움이 된다고 판단하여 ExtraTree 모델을 사용하였다.

각 모델 기준 최고 성능이 나온 **CatBoost, LGBM, ExtraTree**에 대한 **Weighted Average Ensemble**을 진행해주었다.  여러 실험을 해본 결과, LGBM과 Catboost model의 상관관계가 매우 높아 최종 모델로는 **Catboost와 Extratree 두 개의 모델**을 **0.75:0.25 비율**로 설정하여 앙상블을 진행한 것이 성능이 가장 좋아 최종 submission으로 선택하였다.


<br/>

## 5. 팀원 및 담당 역할
**<팀원>**

- 전공생 3명

**<담당 역할>**

- 데이터 전처리 및 피처 엔지니어링
- 모델링 및 하이퍼 파라미터 튜닝
- Weighted Average Ensemble

<br/>

## 6. 자료

- 깃허브  
https://github.com/Ji-eun-Kim/ML-competition-in-kaggle/tree/main


- 리더보드  
https://www.kaggle.com/competitions/kml2022f/leaderboard
<img width="515" alt="머신러닝" src="https://github.com/Ji-eun-Kim/ML-competition-in-kaggle/assets/124686375/ba892dec-f0d9-43c8-9bcd-5fc446ba8178">
<img width="511" alt="머신러닝2" src="https://github.com/Ji-eun-Kim/ML-competition-in-kaggle/assets/124686375/f0d2f04b-cc76-4be1-9a85-d22ebe123677">



































