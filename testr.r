library(readxl)
library(tidyverse)
library(nnet)
library(caret)
#multinomial logistic regression
data <- read_excel("C:/Users/arthu/Downloads/logresmulti.xlsx")

data$Class=data$column_Y
data$New_class <- ifelse(data$Class == 0, "zero",
                 ifelse(data$Class == 1, "one",
                 ifelse(data$Class == 2 , "two",
                      
                        NA)))
data$New_class <- as.factor(data$New_class)
levels(data$New_class)[data(data$New_class) %in% c("zero", "one", "two")] <- "other"
levels(data$New_class)

data$New_class <- relevel(data$New_class, ref = "two")



multinom_model <- multinom(New_class  ~ 
column_0+column_1+column_2+column_3+column_4+column_5 +column_6 +column_7 +column_8 +column_9,
data = data)

summary(multinom_model)

x=data%>%select(column_0,column_1,column_2,column_3,column_4 ,column_5 ,column_6 ,column_7 ,column_8 ,column_9)
predictions <- predict(multinom_model, x, type = "probs")
predictions
# Display the predictions
print(predictions)
1-0.9322702769- 0.0051082386









#binary logistic regression
data <- read_excel("C:/Users/arthu/Downloads/example_LR.xlsx")
data

data$Class=data$y
data$New_class <- ifelse(data$Class == 0, "zero",
                         ifelse(data$Class == 1, "one",
                                
                                       
                                       NA))
data$New_class <- as.factor(data$New_class)
levels(data$New_class)[data(data$New_class) %in% c("zero", "one")] <- "other"
levels(data$New_class)

data$New_class <- relevel(data$New_class, ref = "one")

x=data%>%select(V1,V2,V3,V4,V5)

logistic_model <- glm(New_class ~ V1+ V2 +V3+V4+V5 -1 , data = data, family = binomial,tol)

# Summary of the model
summary(logistic_model)

predictions <- predict(logistic_model, x, type = "response")


