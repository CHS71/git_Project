

library(data.table)

load.libraries <- c('ISLR','plyr', 'dplyr','data.table', 'readxl', 'reshape2', 'stringr','MASS','DMwR', 'stringi', 'ggplot2', 'gridExtra','lubridate','corrplot','e1071','caret','zoo')
install.lib <- load.libraries[!load.libraries %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dependences = TRUE)
sapply(load.libraries, require, character = TRUE)








application <- fread("C:\\DATA\\application.csv",na.strings = c("NA","XNA","XAP",""))






# 라벨 컬럼 분포 확인




## 레이블을 주기위해서 데이터 변형과정을 거친다.

options(scipen=99)

b <-aggregate(SK_ID_CURR~TARGET,application,length)

names(b) <- c("TARGET","Frequency")

rate = round(b$Frequency/sum(b$Frequency)*100,2)
# cbind(b,rate)



ggplot(data=b, aes(x= reorder(TARGET,-Frequency), y=Frequency, fill=factor(TARGET))) +
  geom_bar(stat="identity",width=0.7)+
  geom_text(aes(label= paste(Frequency,"(",rate,"%",")")),size=4, vjust= -0.5)+
  labs(x = " TARGET", y ="Frequency")+
  theme_gray()








## exploratory description analysis


### relation between NAME_CONTRACT_TYPE and TARGET



a <-tapply(application$SK_ID_CURR,list(application$NAME_CONTRACT_TYPE,application$TARGET), length)
a <- data.frame(a)
colnames(a) <- c("repaid","not repaid")
a1 <- data.frame(a,a$`not repaid`/apply(a,1,sum))
colnames(a1)[3] <- c("not repaid ratio")


### integer data column analysis and histogram
z <- sapply(1:dim(application)[2],function(i)class(application[[i]]))

z_num <- which(z == "integer")
application <- data.frame(application)
application_eda <- application[,z_num]


colnames(application_eda)


plotHist <- function(data_in, i) {
  data <- data.frame(x=data_in[[i]])
  p <- ggplot(data=data, aes(x=x)) + geom_histogram(bins=100, fill="#0072B2", alpha = .9) + xlab(colnames(data_in)[i]) + theme_light() +
    theme(axis.text.x = element_text(angle = 90, hjust =1))
  return (p)
}



doPlots <- function(data_in, fun, ii, ncol=3) {
  pp <- list()
  for (i in ii) {
    p <- fun(data_in=data_in, i=i)
    pp <- c(pp, list(p))
  }
  do.call("grid.arrange", c(pp, ncol=ncol))
  
}

doPlots(application_eda, plotHist, ii = 1:12)

doPlots(application_eda, plotHist, ii = 13:24)

doPlots(application_eda, plotHist, ii = 24:35)

doPlots(application_eda, plotHist, ii = 36:41)


###integer형 컬럼 중에서  1 또는 0 ( 범주화시켜야하는 값)을 가지는 컬럼 확인

table_i <- data.frame(colname=colnames(application_eda),category = sapply(1:dim(application_eda)[2],function(i)length(unique(application_eda[[i]])) ))
table_ib <- table_i%>%filter(category < 4)

deleted_col   <- table_ib[[1]]

not_del <- c('TARGET','REGION_RATING_CLIENT_W_CITY','REGION_RATING_CLIENT','REG_CITY_NOT_WORK_CITY',
             'FLAG_EMP_PHONE','REG_CITY_NOT_LIVE_CITY','FLAG_DOCUMENT_3','LIVE_CITY_NOT_WORK_CITY')

delete_col <- unlist(setdiff(deleted_col,not_del))

del_num<- which(colnames(application) %in% (delete_col) )

application <- application[,-del_num]








z <- sapply(1:dim(application)[2],function(i)class(application[[i]]))


### character data column analysis


z_num_c <- which(z == "character")
application <- data.frame(application)
application_eda_c <- application[,z_num_c]
colnames(application_eda_c)

table_c <- data.frame(colname=colnames(application_eda_c),category = sapply(1:dim(application_eda_c)[2],function(i)length(unique(application_eda_c[[i]])) ))


####NAME_CONTRACT_TYPE
a <-tapply(application$SK_ID_CURR,list(application$NAME_CONTRACT_TYPE,application$TARGET), length)
a <- data.frame(a)
colnames(a) <- c("repaid","not repaid")
a1 <- data.frame(a,round(a$`not repaid`/apply(a,1,sum),3)*100)
colnames(a1)[3] <- c("not_repaid_ratio")


####CODE_GENDER
a <-tapply(application$SK_ID_CURR,list(application$CODE_GENDER,application$TARGET), length)
a <- data.frame(a)
colnames(a) <- c("repaid","not repaid")
a1 <- data.frame(a,round(a$`not repaid`/apply(a,1,sum),3)*100)
colnames(a1)[3] <- c("not_repaid_ratio")


####ORGANIZATION_TYPE

a <-tapply(application$SK_ID_CURR,list(application$ORGANIZATION_TYPE,application$TARGET), length)
a <- data.frame(a)
colnames(a) <- c("repaid","not repaid")
a1 <- data.frame(a,round(a$`not repaid`/apply(a,1,sum),3)*100)
colnames(a1)[3] <- c("not_repaid_ratio")

a1 <- a1[order(a1$not_repaid_ratio,decreasing = TRUE),]


####OCCUPATION_TYPE


a <-tapply(application$SK_ID_CURR,list(application$OCCUPATION_TYPE,application$TARGET), length)
a <- data.frame(a)
colnames(a) <- c("repaid","not repaid")
a1 <- data.frame(a,round(a$`not repaid`/apply(a,1,sum),3)*100)
colnames(a1)[3] <- c("not_repaid_ratio")

a1 <- a1[order(a1$not_repaid_ratio,decreasing = TRUE),]






# NA 처리 

##데이터 내의 NA값 비중
mean(is.na(application))

## 변수별 NA값 비중 확인
missing_data <- data.frame(sort(sapply(application, function(x) sum(is.na(x))),decreasing = T))

missing_data <- (missing_data/nrow(application))*100

colnames(missing_data)[1] <- "missingvaluesPercentage"

features <- rownames(missing_data)

missing_data <- cbind(features,missing_data)

rownames(missing_data)<-c(1:nrow(missing_data))

missing_data


## NA값 비중이 50%이상인 컬럼 확인

na60num  <- which(missing_data$missingvaluesPercentage > 50)

missing_data[na60num,]


## NA값 비중이 50%이상인 컬럼 시각화

ggplot(missing_data[missing_data$missingvaluesPercentage > 50,],aes(reorder(features,-missingvaluesPercentage),missingvaluesPercentage,fill= features)) +
  geom_bar(stat="identity") +theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "none") + 
  ylab("Percentage of missingvalues") +
  xlab("Feature") + 
  ggtitle("Understanding Missing Data")


## application 데이터의 NA값 비중이 50%이상인 컬럼인덱스 추출 및 컬럼타입 확인

s <- missing_data$features[missing_data$missingvaluesPercentage > 50]


r <- which(colnames(application) %in% s)

sapply(r, function(i) class( application[[i]] ) )


## application 데이터의 NA값 비중이 50%이상인 컬럼의 카테고리 개수 확인 

cbind(colnames(application)[r] ,sapply(r,function(i) length( unique (application[[i]]) )))


##application 데이터의 NA값 비중이 50%이상인 컬럼삭제

##data.frame으로 변환후 컬럼 삭제.  - > 307511 obs. of  84 variables:
r <- as.numeric(r)
application <- as.data.frame(application)

application_d <- application[,- r]

str(application_d)
#'data.frame':	307511 obs. of  54 variables:

application_d <- na.omit(application_d)


dim(application_d) 
#:  84573     54




##컬럼삭제후 문자형컬럼 카테고리개수 확인 
cn <- sapply(1:dim(application_d)[2],function(i) class(application_d[[i]]))

cnwc <- which(cn=="character")


a_cate_n <- sapply(cnwc,function(i) length( unique (application_d[[i]]) ))




a_cate <- sapply(cnwc, function(i) colnames(application_d)[[i]]  )


cbind(a_cate,cnwc,a_cate_n)



# character -> factor

for (i in cnwc){
  application_d[[i]] <-as.factor(application_d[[i]])
  
}



# integer(1 또는 0 이외의 팩터화시켜야하는 값) -> factor


## integer data column analysis and histogram
z <- sapply(1:dim(application_d)[2],function(i)class(application_d[[i]]))

z_num <- which(z == "integer")
application_d <- data.frame(application_d)
application_eda <- application_d[,z_num]

length(colnames(application_eda)) ## 14

table_i <- data.frame(colname=colnames(application_eda),category = sapply(1:dim(application_eda)[2],function(i)length(unique(application_eda[[i]])) ))


## cumulative density distribution (integer)  


ggplot(application_d, aes(CNT_CHILDREN)) + stat_ecdf(geom = "step", pad = FALSE)


## feature engineering [ CNT_CHILDREN ]  0,1,2  three factor


for (i in 1:dim(application_d)[1]){
  
  if (application_d[i,7] >= 10) {
    
    application_d[i,7] <- as.character(application_d[i,7])
    application_d[i,7] <- gsub(application_d[i,7] , '2' , application_d[i,7])
    application_d[i,7] <- as.numeric(application_d[i,7])
  }
  
  
}

application_d[,"CNT_CHILDREN"] <- as.factor(application_d[,"CNT_CHILDREN"])





# numeric data column correlation analysis 

nume <- sapply(1:dim(application_d)[2],function(i)class(application_d[[i]]))

numeri <- which(nume == "numeric")

numeric_table <- application_d[,numeri]
# numeric_table <- data.frame(application_d$TARGET,numeric_table)

cor_numeric <- cor(numeric_table)


write.csv(cor_numeric,"C:\\DATA\\cor_numeric(3).csv")


##상관성높은 변수제거
ff <- c("AMT_GOODS_PRICE","YEARS_BEGINEXPLUATATION_AVG","YEARS_BEGINEXPLUATATION_MEDI","FLOORSMAX_AVG","YEARS_BEGINEXPLUATATION_MEDI","FLOORSMAX_MEDI","TOTALAREA_MODE","OBS_30_CNT_SOCIAL_CIRCLE","DEF_30_CNT_SOCIAL_CIRCLE")
fff <- c()
for ( i in ff){
  
  sd <- which(colnames(application_d)== i)
  fff <- append(fff,sd)
  
}

application_d <- application_d[,-fff]





#컬럼명구하기
# a <-c()
# 
# 
# for ( i in colnames(application_d)){
#   
#   # print(paste(i, sep = "+"))
#   
#   a <- paste(a,i, sep = "+")
#   
#   
# }
##





# step함수를 사용하여 필요한 변수추출

# # forward regresssion 변수가 너무 많기 때문에 backward보다 forward로 실행한다.
# 
# 
# fit1 <- glm(formula=TARGET~1, data =application_d, family = "binomial")
# fit2 <- glm(formula=TARGET~., data =application_d, family = "binomial")
# step(fit1, direction = "forward",scope=list(upper=fit2,lower=fit1))
# 
# 
# 
# 
# 
# opt_fit <-glm(formula = TARGET ~ EXT_SOURCE_3 + EXT_SOURCE_2 + OCCUPATION_TYPE + 
#                 DAYS_EMPLOYED + FLAG_DOCUMENT_3 + NAME_EDUCATION_TYPE + FLOORSMAX_MODE + 
#                 CODE_GENDER + FLAG_OWN_CAR + NAME_CONTRACT_TYPE + ORGANIZATION_TYPE + 
#                 NAME_FAMILY_STATUS + REGION_RATING_CLIENT_W_CITY + DAYS_ID_PUBLISH + 
#                 DEF_60_CNT_SOCIAL_CIRCLE + NAME_HOUSING_TYPE + DAYS_BIRTH + 
#                 AMT_REQ_CREDIT_BUREAU_QRT + FLAG_OWN_REALTY + AMT_ANNUITY + 
#                 AMT_CREDIT + DAYS_LAST_PHONE_CHANGE + AMT_REQ_CREDIT_BUREAU_MON + 
#                 REGION_POPULATION_RELATIVE + NAME_INCOME_TYPE, family = "binomial", 
#               data = new_train)
# 
# names(unlist(opt_fit[1]))
# 
# opt_col <-c("TARGET","EXT_SOURCE_3","EXT_SOURCE_2","OCCUPATION_TYPE","DAYS_EMPLOYED","FLAG_DOCUMENT_3","NAME_EDUCATION_TYPE","FLOORSMAX_MODE",
#             "CODE_GENDER","FLAG_OWN_CAR","NAME_CONTRACT_TYPE","ORGANIZATION_TYPE","NAME_FAMILY_STATUS","REGION_RATING_CLIENT_W_CITY",
#             "DAYS_ID_PUBLISH","DEF_60_CNT_SOCIAL_CIRCLE","NAME_HOUSING_TYPE","DAYS_BIRTH","AMT_REQ_CREDIT_BUREAU_QRT","FLAG_OWN_REALTY",
#             "AMT_ANNUITY","AMT_CREDIT","DAYS_LAST_PHONE_CHANGE","AMT_REQ_CREDIT_BUREAU_MON","REGION_POPULATION_RELATIVE","NAME_INCOME_TYPE") 
# 
# application_d <- application_d[,opt_col]






# train validataion test split



str(application_d) 
set.seed(1234)
application_d$TARGET <- as.factor(application_d$TARGET)
application_dt <- application_d[,-1]
# application_dt <- application_d

train_cnt <- round(0.6*dim(application_dt)[1]) 


train_index0 <- sample(1:dim(application_dt)[1],train_cnt, replace=F) # 전체 행 숫자 중에서 60% random index 추출  


s <- 1:dim(application_dt)[1] 
r <- setdiff(s, train_index0) # 40% random index



train_cnt1 <- round(0.2*dim(application_dt)[1]) 
train_index1 <- sample(r,train_cnt1, replace=F) # 40% random index 중에서  random index 추출


train_index2 <- setdiff(r,train_index1) # 40% -20% = 나머지 20% random index 추출 



train<- application_dt[train_index0,]
validation <- application_dt[train_index1,]
test <- application_dt[train_index2,]

dim(train)
dim(validation)  
dim(test)


## valid 데이터 설명변수/반응변수 나누기

validation_x <- validation[, -1] 

validation_y <- validation[, 1]





# SMOTE을 사용한 sampling (50:50)

install.packages("DMwR")
library(DMwR)


table(train$TARGET)

new_train <- SMOTE(TARGET ~ ., train, perc.over = 600, perc.under = 140)

table(new_train$TARGET)







#로지스틱 회귀분석 모델 생성 (validation!!)



## model1 :  
model3 <- glm(formula = TARGET ~ ., data = new_train, family = binomial(link='logit'))


summary(model3)  # glm()내에서 자동으로 dummy encoding이 이뤄짐.







#잔차분석
# plot(opt_fit,1)




## full과 step 모델별 확률 예측 
glm.probs3<- predict(model3,newdata= validation_x, type = "response")


glm.probs3[1:5]
length(glm.probs3)

glm.pred3 <- ifelse(glm.probs3 > 0.5, "1", "0")
glm.pred3 <- as.factor(glm.pred3)




##crosstable 로 정확도,오류율 확인 
library(gmodels)
CrossTable(glm.pred3,validation_y)  



##Caret 패키지의 confusionMatrix()함수 사용
library(caret)
confusionMatrix(glm.pred3,validation_y, positive = "1")





##roc curve
install.packages("ROCR")
library(ROCR)

glm.pred3_3 <- prediction(glm.probs3,validation_y)

perf3 <- performance(glm.pred3_3, measure = 'tpr',x.measure = 'fpr')


plot(perf3,main="ROC curve for credit default filter", col='blue',lwd=3)
abline(a=0, b=1, lwd=2, lty=2)


perf.auc3 <- performance(glm.pred3_3, measure = 'auc')
str(perf.auc3)
unlist(perf.auc3@y.values)




















