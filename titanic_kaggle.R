# Introduction and Motivation
# The RMS Titanic was not only the largest ship at that time but also the largest man-made moving object in the world. It was the most deathful/fatal maritime disaster with over 1500 casualties. It is almost %68 of the total passengers and crew. There are several facts about the Titanic disaster. Such as;
# 
# An insufficient number of boats (20 out of 64 boats) that cause this huge number of deaths,
# Most of the survivors being woman and children,
# The most expensive first-class suites priced up to £870 in peak season (£79,000 today) according to Wikipedia.
# Titanic disaster is one of the most important catastrophes that data examination may provide details about survived and non-survived passenger characteristics. Thus, it is crucial to understand survivor profiles in order to save other lives.
# 
# The project have 4 main topics:
#   
# 1. Data Exploration - To understand and get detailed information about dependent variable and other parameters.
# 2. Feature Engineering - To create new parameters that can increased accuracy and help to make better prediction.
# 3. Data Analysis - To prepeare and fit the parameters for building model
# 4. Model Building - To build different models and comparing their accuracy for making prediction with the best of.


library(ggplot2)
library(tidyverse)
library(knitr)
library(corrplot)
library(caret)
library(randomForest)
library(gbm)
library(vcd)
library(varSelRF)
library(ROCR)


train_data <- read.csv("../input/train.csv")
test_data <- read.csv("../input/test.csv")


#1. Exploratory Data Analysis
# Looking underlying structure the structure and dimension of train data

str(train_data)
dim(train_data)

# Printing first rows of train data
head(train_data, n = 10)


# Checking if there is any missing data

colSums(is.na(train_data))


#if there is any invalid data like negative values or extra category in any parameters .

summary(train_data)

#Another part of data exploratory is predictors' effect on survivals. 
#I will summarize the parameters mostly by visualizing them. 
#It will help to discover patterns, identify important variables, make statistical summary, outliers control etc. 
#Some graphs created for getting more knowledge about survived passenger specifics.
#First, Survived group is converted to factor to be able to achieve better visual results

train_data$Survived <- as.factor(train_data$Survived)

## Survived vs Sex

#In Titanic facts, it is said that most of the rescued people were women.
#We can create a graph to visualize our data results. 
#For categorical data, bar graph is the most common way for visualization


ggplot(data = train_data, aes(x = Survived, fill = Sex)) +
  geom_bar() +
  geom_text(aes(label = scales::percent(..count.. / sum(..count..))), stat = 'count', position = position_stack(0.5)) +
  ggtitle("Survived vs Sex") + xlab("Survived") + ylab ("Count") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

## Survived vs Age

# Age is a continuous variable and it is not able to show survival rates on specific age ranges with continuous format.
# In general, the conversion continuous variables to categorical type can cause loss of information.
# However, in this case, the conversion to categorical variable makes interpretation simpler.
# Age variable split by 18 for range.
# This number is chosen to be able to seperate well children, young, adult and older persons.

train_data$Grouping_Age <-
  cut(train_data$Age, breaks <- c(seq(0, 100, by <- 18), Inf))

ggplot(data = train_data, aes(x = Grouping_Age, y = Survived)) +
  geom_bar(aes(fill = Survived), stat = "identity") +
  scale_fill_brewer(palette = "Dark2") +
  ggtitle("Survived vs Age") + xlab("Age Range") + ylab ("Survived or No Survived") +
  theme_classic() + coord_flip() +
  theme(plot.title = element_text(hjust = 0.5))

## Survived vs Ticket Clas

#Another graph is about Pclass vs Survived. First pclass convert to factor

train_data$Pclass <- as.factor(train_data$Pclass)

ggplot(data = train_data, aes(x = Pclass, y = Survived)) +
  geom_bar(aes(fill = Survived), stat = "identity") +
  scale_fill_brewer(palette = "Reds") +
  labs(title = "Ticket Class vs Survived", x = "Ticket Class", y = "Survived or No Survived") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))

## Survived vs Embarked

# 2. Feature Engineering

# First, I want to create total number of family features. Below graphs shows that, in both group most passengers are single.
#This features will help to analyze single passenger more easier and I may achieved more significant results.

d1 = ggplot(data = train_data, aes(Parch)) +
  geom_bar(stat = "Count") +
  theme_minimal()

d2 = ggplot(data = train_data, aes(SibSp)) +
  geom_bar(stat = "Count") +
  theme_minimal()

grid.arrange(d1, d2, nrow = 1)

#Creating Total_Family
train_data$Total_Family <- train_data$SibSp + train_data$Parch + 1

#Below table shows that most survived and non survived persons have less family member

train_data %>%
  group_by(Survived, Total_Family) %>%
  summarise_(n = ~ n()) %>%
  mutate(prop = prop.table(n)) %>%
  kable()

#Grouping_Age vs Total_Family Graph

#Converting factor to visualize it
train_data$Total_Family = as.factor(train_data$Total_Family)

ggplot(data = train_data, aes(x = Grouping_Age, fill = Total_Family)) +
  geom_bar() +
  geom_text(stat = 'count', aes(label = ..count..),
            position = position_stack(vjust = 0.5), size = 2) +
  labs(title = "Age vs Total_Family", x = "Age", y = "Count") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5))


#Convert back to numeric 
train_data$Total_Family = as.numeric(train_data$Total_Family)

# Second new parameter is Ticket Number. The aim of this engineering is creating group passenger and adding to model.

train_data$Ticket_Code <- gsub('\\s[0-9]+|[0-9]{2,}|\\.', "", train_data$Ticket)

train_data$Ticket_Number <- gsub('\\D+', '', train_data$Ticket)

#Create new table for ticket number size
Ticket_Number_Size <-
  train_data %>%
  group_by(Ticket_Number) %>%
  tally()

#Converting to Data Frame
Ticket_Number_Size = as.data.frame(Ticket_Number_Size)

#Join both table and adding to train_data
train_data =
  train_data %>%
  inner_join(Ticket_Number_Size, by = "Ticket_Number")

train_data <-
  rename(train_data, Ticket_Size = n)

#Add ticket number for Line tickets which seems empty value on the table.
train_data =
  train_data %>%
  mutate(Ticket_Number = ifelse(Ticket_Number == "", 0000 , Ticket_Number))

#Adding Grouping Column
#0 = Non-Group Passenger
#1 = Grouping Passenger

train_data <-
  train_data %>%
  mutate(Grouping_Ticket = ifelse(Ticket_Size == 1, 0, 1))

train_data$Grouping_Ticket = as.factor(train_data$Grouping_Ticket)

#Converting Factor 
train_data$Grouping_Ticket <- as.factor(train_data$Grouping_Ticket)

#3. Data Analysis

#Let's do the replacement of missing values. Age variables have around 177 null values. 
#This values replace with its median. This methods work on randomly missing values

train_data <-
  train_data %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age))


#Embarked table shows that, there is 2 empty values. The proportion of empty values are very low (< 0.1),
#so both values can be replaced with only "S" or "S" and "C" which are categories that have highest values.
#In this case, these values will be replaced with only "S"
train_data =
  train_data %>%
  mutate(Embarked = ifelse(Embarked == "", 4 , Embarked))

train_data$Embarked = as.factor(train_data$Embarked)


#Fare variable is continues. So, first we can check the range, distribution or etc.

range(train_data$Fare)

#The range gap seems so wide, we can look details with the density graph

ggplot(data = train_data,
       aes(x = Fare, fill = Survived),
       binwidth = bwidth) +
  geom_area(stat = "bin" , binwidth = 25) +  geom_density(alpha = .2) + theme_classic() + scale_fill_brewer(palette =
                                                                                                              "Paired") +
  labs(title = "Fare Density", x = "Fare", y = "Count") + theme(plot.title = element_text(hjust = 0.5))

# Also some passenger's ticket price is zero.

#First examine them
train_data %>%
  filter(Fare == 0)

#Second analysis about survival behaviour on Fare parameters. To achieve this, Fare is converted to categorical values. 
#Let's create seperate categories for the min and max fares. 
#Also the distribution is dense on the range of 1 to 100.

train_data = train_data %>%
  mutate(Categorical_Fare = cut(
    Fare,
    breaks = c(-Inf, 0, 10, 50, 100, 200, 300, 600),
    labels = c("0", "1-10", "11-50", "51 - 100", "101-200", "201-300", "500+")
  ))

train_data$Categorical_Fare = as.factor(train_data$Categorical_Fare)

##Visualize
ggplot(data = train_data, aes(x = Categorical_Fare, y = Survived)) +
  geom_bar(aes(fill = Survived), stat = "identity") + scale_fill_brewer(palette =
                                                                          "Accent") + theme_classic() +
  labs(title = "Fare Category vs Survived", x = "Fare Category", y = "Survived or No Survived")

###############
### Correlation

#Another step is correlation coefficients between variables. 
#For Contunies variables, correlation matrix is one of the best method.
#But this data have a lot of categorical and few continuse data.
#Correlation matrix will build with continues data to explore positive or negative relationship between them with Pearson correlation method.

numeric_data = train_data[, c(6:8, 10, 14, 17)]
corr_matrix <- cor(numeric_data, method = "pearson")

corrplot.mixed(corr_matrix, tl.cex = .6, addrect = 2)

#Secondly, measure of strength of the association between categorical parameters will be checked.
#In this circumstances, Person Chi-Square can used for significance test and Cramer's V can used for effect size of parameters.
#These test can help to choose which parameter will be used on model building.

#First some categorical parameters will be visualizing.

mosaic(
  ~ Pclass + Sex + Grouping_Ticket + Survived,
  data = train_data,
  gp = shading_hcl,
  main = "Mosaic Plot for Categorical Variables"
)


#For Person Chi-Square, first hypothesizes.

#H0: The two variables have not significance relationship
#H1: The two variables have significance relationship


table1 = xtabs(~ Survived + Pclass, data = train_data)
assocstats(table1)


#p-Value of Pearson(0) less than the significance level of 0.05. It means null hypotesis rejected. It bring to a conclusion that there is a significant relationship between them.
# and the relationship between them is strong based on the Cramer's V result(0.34).

table2 = xtabs(~ Survived + Sex, data = train_data)
assocstats(table2)

#p-Value of Pearson(0) less than the significance level of 0.05. It means null hypotesis rejected. It bring to a conclusion that there is a significant relationship between them.
# and the relationship between them is very strong based on the Cramer's V result.


table3 = xtabs(~ Survived + Embarked, data = train_data)
assocstats(table3)
#p-Value of Pearson(0) less than the significance level of 0.05. It means null hypotesis rejected. It bring to a conclusion that there is a significant relationship between them.
# and the relationship between them is weak based on the Cramer's V result.

table4 = xtabs(~ Survived + Grouping_Ticket, data = train_data)
assocstats(table4)
#p-Value of Pearson(0) less than the significance level of 0.05. It means null hypotesis rejected. It bring to a conclusion that there is a significant relationship between them.
# and the relationship between them is weak based on the Cramer's V result.

table5 = xtabs(~ Survived + Categorical_Fare, data = train_data)
assocstats(table5)
#p-Value of Pearson(0) less than the significance level of 0.05. It means null hypotesis rejected. 

#Final step of Train Data analysis is creating new data frame with parameters which they will used on the model.
#The parameters are: Survived, Pclass, Sex, Age, Embarked, Total_Family, Grouping_Ticket and Categorical_Fare

train_data_model = train_data[, c(2:3, 5:6, 12, 14, 18:19)]



####################
######TEST DATA ####
####################


# Lets look out the test data quickly and add new features the same of train data

str(test_data)

#Convert Pclass to factor
test_data$Pclass = as.factor(test_data$Pclass)

#Convert Embaked names same with Train Data
test_data <- test_data %>%
  mutate(Embarked = recode(Embarked, C = "2",
                           Q = "3",
                           S = "4"))

test_data %>%
  group_by(Embarked) %>%
  summarize(n = n()) %>%
  kable()

# Checking is there a missing data

colSums(is.na(test_data))

#It seems that, age data and fare data have missing values. 
#Fare have only 1 missing values. Both group will be replace with median.

test_data =
  test_data %>%
  mutate(Age = ifelse(is.na(Age), median(Age, na.rm = TRUE), Age))

test_data =
  test_data %>%
  mutate(Fare = ifelse(is.na(Fare), median(Fare, na.rm = TRUE), Fare))

##
#Create Total Family for test data

test_data$Total_Family = test_data$SibSp + test_data$Parch + 1

##
#Create Grouping Ticket parameter for test data

test_data$Ticket_Number = gsub('\\D+', '', test_data$Ticket)


#Create new table for ticket number size
Ticket_Number_Size <-
  test_data %>%
  group_by(Ticket_Number) %>%
  tally()

#Converting to Data Frame
Ticket_Number_Size = as.data.frame(Ticket_Number_Size)

#Join both table and adding to train_data
test_data =
  test_data %>%
  inner_join(Ticket_Number_Size, by = "Ticket_Number")

test_data <-
  rename(test_data, Ticket_Size = n)

#Add ticket number for Line tickets which seems empty value on the table.
test_data =
  test_data %>%
  mutate(Ticket_Number = ifelse(Ticket_Number == "", 0000 , Ticket_Number))

#Adding Grouping Column
#0 = Non-Group Passenger
#1 = Grouping Passenger

test_data <-
  test_data %>%
  mutate(Grouping_Ticket = ifelse(Ticket_Size == 1, 0, 1))

test_data$Grouping_Ticket = as.factor(test_data$Grouping_Ticket)

table(test_data$Grouping_Ticket)

#Converting Fare to categorical data
test_data = test_data %>%
  mutate(Categorical_Fare = cut(
    Fare,
    breaks = c(-Inf, 0, 10, 50, 100, 200, 300, 600),
    labels = c("0", "1-10", "11-50", "51 - 100", "101-200", "201-300", "500+")
  ))

#Create total_family group for test data

test_data$Total_Family = test_data$SibSp + test_data$Parch + 1

#Converting Fare to categorical data
test_data_1 = test_data %>%
  mutate(Categorical_Fare = cut(
    Fare,
    breaks = c(-Inf, 0, 10, 50, 100, 200, 300, 600),
    labels = c("0", "1-10", "11-50", "51 - 100", "101-200", "201-300", "500+")
  ))

test_data_1$Categorical_Fare = as.factor(test_data_1$Categorical_Fare)

#Choosing parameters for test_data
test_data_model = test_data[, c(2, 4:5, 11:12, 15:16)]

str(train_data_model)
str(test_data_model)


# 4. Model Building

# 3 Different Supervised Learning Algorithms will be used for model building: SVM, Random Forest and Gradient Boosting.
## SVM will be used because it is one of the powerful statistical technique for binary classification.
## Random forest will be used because most of the dependence variables are categorical and random forest work well with them.
## Gradient Bossting method will be used. It build trees one at a time, where each new tree helps to correct errors made by previously trained tree while RF train each tree independently, using a random sample of the data.
## Extreme Gradient Boosting method choosed. It performs parallel processing to improve computational efficiency.

### Choosing important Parameters
# Hyperparameter optimization approach implemented before training model build. This approach will help to select optimal hyperpatameters or in other word tune parameters. 
# Thus, loss function will be minimize. 
# Gradient Boosting method will help to find relative influence for each parameters. 
# Gradient Boosting: GBT build trees one at a time, where each new tree helps to correct errors made by previously trained tree.

GBM_model = gbm(Survived ~ ., 
           data = train_data_model, distribution = "multinomial",
           bag.fraction = 0.5, n.trees = 1000, interaction.depth =3, shrinkage = 0.01, n.minobsinnode = 10, cv.folds = 10)

par(mar = c(5, 8, 1, 1))
summary(
 GBM_model, 
  cBars = 10,
  method = relative.influence, # also can use permutation.test.gbm
  las = 2
)

#Random Forest Method with varSelRF library
RF_variables= varSelRF(train_data_model[, -1], train_data_model[, 1], ntree = 1000,
                       c.sd=1, returnFirstForest = TRUE)

print(RF_variables)

#The result as same as with Gradient Boosting Method. On the model, "Age", "Sex", "Pclass", "Categorical Fare" and Total Family parameters will be used. 

train_data_model = train_data_model[, c(1:4, 6, 8)]

### Data Splitting

#Train data will split to train(70%) and test data(30%). The aim of this splitting is minimizing the error rate
#Train set used for model building and test set used for model assess.
#Another important thing is cross validation. This method will help to measure and assesing model performance and avoid overfitting.

trainIndex = createDataPartition(y=train_data_model$Survived, p=0.70, list=FALSE)

train_set = train_data_model[trainIndex,]
test_set = train_data_model[-trainIndex,]

#10 times repeated 10 fold cross validation will be used.
set.seed(123)
trCont = trainControl(method = "repeatedcv",
                      number = 10,
                      repeats = 10)

### SVM with Radial Basis Function Kernel

SVM_model = train(
  Survived ~ .,
  data = train_set,
  method = "svmRadial",
  trControl = trCont
)


print(SVM_model)
#Prediction
SVM_pred <- predict(SVM_model, test_set)
#Confusion Matrix
SVM_Conf <- confusionMatrix(SVM_pred, test_set$Survived)
print(SVM_Conf)

#Random Forest
#First RandomForest package will be used and check Gini coefficient for variable importance

RF_model = randomForest(Survived ~ .,
                        data = train_set,
                        importance = TRUE,
                       ntree = 1000)
print(RF_model) 
varImpPlot(RF_model)
importance(RF_model)

## Cross validation method.

## Cross validation method.

RF_model_caret <- train(
  Survived ~  .,
  data = train_set,
  method = "rf",
  trControl = trCont
)

#Prediction
RF_pred <- predict(RF_model_caret, test_set)

#Confusion Matrix
RF_Conf <- confusionMatrix(RF_pred, test_set$Survived)
print(RF_Conf)

## Extreme Gradient Boosting

XGBM_model = train(
  Survived ~  .,
  data = train_set,
  method = "xgbTree",
  trainControl = trCont
)

print(XGBM_model)
#Prediction
XGBM_pred <- predict(XGBM_model, test_set)

#Confusion Matrix
XGBM_Conf <- confusionMatrix(XGBM_pred, test_set$Survived)
print(XGBM_Conf)



# Comparing Machine Learning Models with ROC

pred_combine <- cbind(SVM_pred, RF_pred, XGBM_pred)

prediction_matrix <- prediction(pred_combine, labels = matrix(test_set$Survived, 
                                                              nrow = length(test_set$Survived), ncol = 3) )

performance_matrix <- performance(prediction_matrix, "tpr", "fpr")
plot(performance_matrix, col = as.list(1:3), main = "ML-ROC Curves", 
     type = "l",xlab= "False Positive Rate", ylab="True Positive Rate")
legend(x = "bottomright", legend = c("SVM", "RF", "GBM"), fill = 1:3)