#종속변수1: 수면
setwd('C:/Users/kim ji eun/Desktop/2학년 과목/회귀분석/기말')
df<- read.csv('df1.csv',header=T)
head(df)
tail(df)

Y <- df[,1]
X1 <- df[,2]
X2 <- df[,3]
X3 <- df[,4]
X4 <- df[,5]
X5 <- df[,6]
X6 <- df[,7]
X7 <- df[,8]

#############################################################################

# 탐색적 분석
# scatter plot
#X1(아침식사 빈도)
par(mfrow=c(1,1))
plot(x=X1,y=Y,pch=19,cex=2,xlab="X1(아침식사 빈도)",ylab="수면")
model1 <- lm(Y~X1);model1
abline(model1,col="red",lwd=2)

#X2(한 번에 마시는 음주량_잔)
par(mfrow=c(1,1))
plot(x=X2,y=Y,pch=19,cex=2,xlab="X2(한 번에 마시는 음주량_잔)",ylab="수면")
model2 <- lm(Y~X2);model2
abline(model2,col="red",lwd=2)

#X3(평소 하루 앉아서 보내는 시간)
par(mfrow=c(1,1))
plot(x=X3,y=Y,pch=19,cex=2,xlab="X3(평소 하루 앉아서 보내는 시간)",ylab="수면")
model3 <- lm(Y~X3);model3
abline(model3,col="red",lwd=2)

#X4(하루평균 흡연량)
par(mfrow=c(1,1))
plot(x=X4,y=Y,pch=19,cex=2,xlab="X4(하루평균 흡연량)",ylab="수면")
model4 <- lm(Y~X4);model4
abline(model4,col="red",lwd=2)

#X5(1일 당 섭취량)
par(mfrow=c(1,1))
plot(x=X5,y=Y,pch=19,cex=2,xlab="X5(1일 당 섭취량)",ylab="수면")
model5 <- lm(Y~X5);model5
abline(model5,col="red",lwd=2)

#X6(1일 탄수화물 섭취량)
par(mfrow=c(1,1))
plot(x=X6,y=Y,pch=19,cex=2,xlab="X6(1일 탄수화물 섭취량)",ylab="수면")
model6 <- lm(Y~X6);model6
abline(model6,col="red",lwd=2)

#X7(걷기 지속 시간)
par(mfrow=c(1,1))
plot(x=X7,y=Y,pch=19,cex=2,xlab="X7(걷기 지속 시간)",ylab="수면")
model7 <- lm(Y~X7);model7
abline(model7,col="red",lwd=2)

#############################################################################


# (1) Full model
Full_model <- lm(Y ~ X1 + X2 + X3+ X4+ X5 +X6 +X7)
summary(Full_model)


# full model 다중공선성
diag(solve(R))
install.packages('car')
library("car")
vif(Full_model)


# Full modedl에 대한 그래프 확인
par(mfrow=c(2,2))
plot(Full_model)


#############################################################################


#변수선택
df_rm<-read.csv('df1.csv',header=T)
df_rm

# 변수선택(full model_adj_R2)
library(leaps)
qustn <- regsubsets(x=수면~.,data=df,method="exhaustive",nbest = 7)
summary(qustn)
result_regfit <- summary(qustn)
result_regfit$adjr2 # adj_R2


# plot of adj_R2
par(mfrow=c(1,1))
plot(result_regfit$adjr2,pch=19,cex=2,ylab="adj_R2",xlab="model",type="b")


#변수 6개에 대한 검정
qustn6<- lm(Y ~ X1 + X2 + X3+ X4+ X5 +X7)
summary(qustn6)


#null_model 과 ANOVA분석
null_model <- lm(Y~1)
anova(qustn6,null_model)


# Mallows_Cp
result_regfit$cp 


# plot of Mallows-Cp
plot(result_regfit$cp,pch=19,cex=2,ylab="Mallows-Cp",xlab="model",type="b")

#변수 1개에 대한 검정
qustn1<- lm(Y ~X4 )
summary(qustn1)


#null_model 과 ANOVA분석
null_model <- lm(Y~1)
anova(qustn1,null_model)


#다중공선성
diag(solve(R))
install.packages('car')
library("car")
vif(qustn6)

#############################################################################

# 단계별 회귀검정
null_model <- lm(Y~1)
step(null_model,scope = ~ X1 +X2 +X3 +X4 +X5 +X7,direction="both",test="F")

#############################################################################

#변수변환

#boxcox 변환
par(mfrow=c(1,1))
library(MASS)
boxcox(qustn6)


#Y
Full_ <- lm(Y~ X1 +X2 +X3 +X4 +X5 +X7)
par(mfrow=c(2,2))
plot(Full_)
summary(Full_)

#log 변환
log_Rm_model <- lm(log(Y) ~ X1 + X2 + X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(log_Rm_model)
summary(log_Rm_model)


#루트 변환
sqrt_Rm_model <- lm(sqrt(Y) ~ X1 + X2 + X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(sqrt_Rm_model)
summary(sqrt_Rm_model)


#1/Y 변환 
r_Rm_model <- lm(1/Y ~ X1 + X2 + X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(r_Rm_model)
summary(r_Rm_model)

############################################################################

#Y
Final_model <- lm(log(Y)~ X1 +X2 +X3 +X4 +X5 +X7)
par(mfrow=c(2,2))
plot(Final_model)
summary(Final_model)

Final_model


# BIC
result_regfit$bic
#############################################################################


#회귀모형의 선택

# SST, SSE, SSR
SST <- sum((df$'수면'-mean(df$'수면'))^2)
SSE <- sum(resid(Final_model)^2)
SSR <- SST-SSE

#PRESS
#SSE값
Final_model <- lm(log(Y)~ X1 +X2 +X3 +X4 +X5 +X7)
sum(resid(Final_model)^2)
#PRESS값
press<-sum((resid(Final_model)/(1-hatvalues(Final_model)))^2)


# R2 vs R2_predic
1-(SSE/SST) # R2 = 0.995034
1-(press/SST) # R2_predic = 0.9949024

##############################################################################
#종속변수2: 월평균 가구총소득

setwd('C:/Users/kim ji eun/Desktop/2학년 과목/회귀분석/기말')
df<- read.csv('df2.csv',header=T)
head(df)
tail(df)

Y <- df[,1]
X1 <- df[,2]
X2 <- df[,3]
X3 <- df[,4]
X4 <- df[,5]
X5 <- df[,6]
X6 <- df[,7]
X7 <- df[,8]

###########################################################################

# 탐색적 분석
# scatter plot
#X1(아침식사 빈도)
par(mfrow=c(1,1))
plot(x=X1,y=Y,pch=19,cex=2,xlab="X1(아침식사 빈도)",ylab="월평균 가구총소득")
model1 <- lm(Y~X1);model1
abline(model1,col="red",lwd=2)

#X2(한 번에 마시는 음주량_잔)
par(mfrow=c(1,1))
plot(x=X2,y=Y,pch=19,cex=2,xlab="X2(한 번에 마시는 음주량_잔)",ylab="월평균 가구총소득")
model2 <- lm(Y~X2);model2
abline(model2,col="red",lwd=2)

#X3(평소 하루 앉아서 보내는 시간)
par(mfrow=c(1,1))
plot(x=X3,y=Y,pch=19,cex=2,xlab="X3(평소 하루 앉아서 보내는 시간)",ylab="월평균 가구총소득")
model3 <- lm(Y~X3);model3
abline(model3,col="red",lwd=2)

#X4(하루평균 흡연량)
par(mfrow=c(1,1))
plot(x=X4,y=Y,pch=19,cex=2,xlab="X4(하루평균 흡연량)",ylab="월평균 가구총소득")
model4 <- lm(Y~X4);model4
abline(model4,col="red",lwd=2)

#X5(1일 당 섭취량)
par(mfrow=c(1,1))
plot(x=X5,y=Y,pch=19,cex=2,xlab="X5(1일 당 섭취량)",ylab="월평균 가구총소득")
model5 <- lm(Y~X5);model5
abline(model5,col="red",lwd=2)

#X6(1일 탄수화물 섭취량)
par(mfrow=c(1,1))
plot(x=X6,y=Y,pch=19,cex=2,xlab="X6(1일 탄수화물 섭취량)",ylab="월평균 가구총소득")
model6 <- lm(Y~X6);model6
abline(model6,col="red",lwd=2)

#X7(걷기 지속 시간)
par(mfrow=c(1,1))
plot(x=X7,y=Y,pch=19,cex=2,xlab="X7(걷기 지속 시간)",ylab="월평균 가구총소득")
model7 <- lm(Y~X7);model7
abline(model7,col="red",lwd=2)

###############################################################################

# (1) Full model
Full_model <- lm(Y ~ X1 + X2 + X3+ X4+ X5 +X6 +X7)
summary(Full_model)


# full model 다중공선성
diag(solve(R))
#install.packages('car')
library("car")
vif(Full_model)


# Full modedl에 대한 그래프 확인
par(mfrow=c(2,2))
plot(Full_model)

Full_model

##############################################################################
#변수선택
df_rm<-read.csv('df2.csv',header=T)
df_rm


# 변수선택(full model_adj_R2)
library(leaps)
qustn <- regsubsets(x=월평균.가구총소득~.,data=df,method="exhaustive",nbest = 7)
summary(qustn)
result_regfit <- summary(qustn)
result_regfit$adjr2 # adj_R2


# plot of adj_R2
par(mfrow=c(1,1))
plot(result_regfit$adjr2,pch=19,cex=2,ylab="adj_R2",xlab="model",type="b")


#변수 4개에 대한 검정
qustn4<- lm(Y ~ X3+X4+ X5+X7)
summary(qustn4)


#null_model 과 ANOVA분석
null_model <- lm(Y~1)
anova(qustn4,null_model)


# Mallows_Cp
result_regfit$cp 


# plot of Mallows-Cp
plot(result_regfit$cp,pch=19,cex=2,ylab="Mallows-Cp",xlab="model",type="b")


#변수 4개에 대한 검정
qustn4<- lm(Y ~ X3+ X4 + X5+ X7)
summary(qustn4)

#변수 2개에 대한 검정
qustn2<- lm(Y~ X4+X5)
summary(qustn2)


#null_model 과 ANOVA분석
null_model <- lm(Y~1)
anova(qustn4,null_model)


#다중공선성2
diag(solve(R))
install.packages('car')
library("car")
vif(qustn4)


##############################################################################
#단계별 회귀
null_model <- lm(Y~1)
step(null_model,scope = ~X3 +X4 +X5 +X7,direction="both",test="F")

##############################################################################

#변수변환

#boxcox 변환
par(mfrow=c(1,1))
library(MASS)
boxcox(qustn4)



#Y
Full_ <- lm(Y~ X3 +X4 +X5 +X7)
par(mfrow=c(2,2))
plot(Full_)
summary(Full_)

#log 변환
log_Rm_model <- lm(log(Y) ~ X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(log_Rm_model)
summary(log_Rm_model)


#루트 변환
sqrt_Rm_model <- lm(sqrt(Y) ~ X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(sqrt_Rm_model)
summary(sqrt_Rm_model)

#1/Y 변환
r_Rm_model <- lm(1/(Y) ~ X3+ X4+ X5 +X7)
par(mfrow=c(2,2))
plot(r_Rm_model)
summary(r_Rm_model)


#BIC 검정
#BIC
par(mfrow=c(1,1))
result_regfit$bic

# plot of BIC
plot(result_regfit$bic,pch=19,cex=2,ylab="BIC",xlab="model",type="b")

#Y
Final_model <- lm(sqrt(Y)~ X3 +X4 +X5 +X7)
par(mfrow=c(2,2))
plot(Final_model)
summary(Final_model)
Final_model

##############################################################################

#회귀모형의 선택

# SST, SSE, SSR
SST <- sum((df$'월평균.가구총소득'-mean(df$'월평균.가구총소득'))^2)
SSE <- sum(resid(Final_model)^2)
SSR <- SST-SSE

#PRESS
#SSE값
Final_model <- lm(sqrt(Y)~ X1 +X2 +X3 +X4 +X5 +X7)
sum(resid(Final_model)^2)

#PRESS 값
press<- sum((resid(Final_model)/(1-hatvalues(Final_model)))^2)
press

# R2 vs R2_predic
1-(SSE/SST) # R2 = 0.9994998
1-(press/SST) # R2_predic = 0.9994911


Final_model

