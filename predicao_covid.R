# Ambiente ----------------------------------------------------------------
options(scipen=999)
gc()
set.seed(1)

# Pacotes -----------------------------------------------------------------
library(readr)
library(tidyverse)
library(reshape2)
library(caret)
library(mlr)
library(forecast)
library(doParallel)
library(parallelMap)
library(ggpubr)


# Importação da base ------------------------------------------------------
covid <- read_csv("dados/COVID2.csv", locale = locale(encoding = "WINDOWS-1252"))

#Retirando caracteres especiais
rm_accent <- function(str,pattern="all") {
  if(!is.character(str))
    str <- as.character(str)

  pattern <- unique(pattern)

  if(any(pattern=="Ç"))
    pattern[pattern=="Ç"] <- "ç"

  symbols <- c(
    acute = "áéíóúÁÉÍÓÚýÝ",
    grave = "àèìòùÀÈÌÒÙ",
    circunflex = "âêîôûÂÊÎÔÛ",
    tilde = "ãõÃÕñÑ",
    umlaut = "äëïöüÄËÏÖÜÿ",
    cedil = "çÇ"
  )

  nudeSymbols <- c(
    acute = "aeiouAEIOUyY",
    grave = "aeiouAEIOU",
    circunflex = "aeiouAEIOU",
    tilde = "aoAOnN",
    umlaut = "aeiouAEIOUy",
    cedil = "cC"
  )

  accentTypes <- c("´","`","^","~","¨","ç")

  if(any(c("all","al","a","todos","t","to","tod","todo")%in%pattern)) # opcao retirar todos
    return(chartr(paste(symbols, collapse=""), paste(nudeSymbols, collapse=""), str))

  for(i in which(accentTypes%in%pattern))
    str <- chartr(symbols[i],nudeSymbols[i], str)

  return(str)
}


covid <- sapply(covid, rm_accent)%>% as.data.frame()
covid <- sapply(covid, tolower)%>% as.data.frame()

covid <- subset(covid, !is.na(covid$DT_NOT))



covid$DT_NOT <- as.Date(covid$DT_NOT, format = "%d/%m/%Y")
covid$DN <- as.Date(covid$DN, format = "%d/%m/%Y")
covid$TRIAGEM <- ifelse(covid$DT_NOT < as.Date("24/03/2020", format = "%d/%m/%Y"), "modelo 1", "modelo 2")
covid$IDADE <- (covid$DT_NOT - covid$DN)/365
covid$IDADE <- as.numeric(as.character(covid$IDADE)) 
covid[which(covid$IDADE < 0.001), colnames(covid) == "IDADE"] <- NA
covid$DN <- NULL
covid$DT_NOT <- NULL
covid$BAIRRO<- NULL
#Substituindo na por "missing" nas variáveis que são fator. Com a idade não será feito isso, pois é uma variável numérica
covid <- subset(covid, !is.na(covid$IDADE))
covid[,-c(12)] <- sapply(covid[,-c(12)], as.factor) %>% as.data.frame()
for(i in 1:length(names(covid[,-c(12)]))){
	covid[,i] <-fct_explicit_na(covid[,i], "missing")
}


summary(covid)

covid$OCUPACAO <- droplevels(covid$OCUPACAO)
covid$BAIRRO <- droplevels(covid$BAIRRO)
covid$MUNICIPIO <- droplevels(covid$MUNICIPIO)


#Criando base de treino e base de teste. Os inconclusivos foram tirados
train_base <- subset(covid, covid$RESULTADO == "descartado" |
	       	      covid$RESULTADO == "confirmado")
train_base$RESULTADO <- factor(train_base$RESULTADO, levels = c("confirmado", "descartado"))
summary(train_base)


predic_base <- subset(covid, covid$RESULTADO == "missing")
predic_base$RESULTADO <- NULL
summary(predic_base)



#Rodando Modelo com MLR
parallelStartSocket(4)
mod1_task <- makeClassifTask(data = train_base[,-1], target = "RESULTADO")
mod1_task_over <- oversample(mod1_task, rate = 3) #Tem 3 descartado para 1 confirmado
mod1_task_under <- undersample(mod1_task, rate = 1/3) # Tem 1 confirmado para 3 descartados
mod1_task_smote <- smote(mod1_task, rate = 3, nn = 5)

lrns_type <- c('classif.adaboostm1',
		'classif.bartMachine',
		'classif.boosting',
		'classif.gamboost',
		'classif.gbm',
		'classif.glmboost',
		'classif.glmnet',
		'classif.h2o.deeplearning',
		'classif.h2o.gbm',
		'classif.h2o.glm',
		'classif.h2o.randomForest',
		'classif.naiveBayes',
		'classif.randomForest',
		'classif.randomForestSRC',
		'classif.ranger',
		'classif.svm')


lrns <- list()
for(i in seq_along(lrns_type)){
	lrns[[i]] <- lrns_type[i]
}


cross_val <- makeResampleDesc("CV", iter = 5)
benc <-  benchmark(tasks = mod1_task, learners = lrns, resampling = cross_val, 
		   measures = list(tnr), show.info = FALSE, models = TRUE)
benc_under <-  benchmark(tasks = mod1_task_under, learners = lrns, resampling = cross_val, 
			 measures = list(tnr), show.info = FALSE, models = TRUE)
benc_over <-  benchmark(tasks = mod1_task_over, learners = lrns, resampling = cross_val, 
			measures = list(tnr), show.info = FALSE, models = TRUE)
benc_smote <-  benchmark(tasks = mod1_task_smote, learners = lrns, resampling = cross_val, 
			 measures = list(tnr), show.info = FALSE, models = TRUE)

plotBMRBoxplots(benc, measure =tnr)
plotBMRBoxplots(benc_under, measure =tnr)
plotBMRBoxplots(benc_over, measure =tnr)
plotBMRBoxplots(benc_smote, measure =tnr)



#Analisando o modelo selecionado
set.seed(1)
result_train <- resample(learner = "classif.h2o.randomForest", task = mod1_task_smote, resampling = cross_val, show.info = FALSE)
confusionMatrix(data = result_train$pred$data$response, reference = result_train$pred$data$truth)
confusionMatrix(data = result_train$pred$data$response, reference = result_train$pred$data$truth, mode = "prec_recall")

#Utilizando o modelo selcionado
mod_pred <- train(mod1_task_smote, learner = "classif.h2o.randomForest")



parallelStop()

#Fazendo o gráfico
predic_base$RESULTADO <- predict(mod_pred, newdata = predic_base[,-1])$data[,1]
predic_base$RESULTADO <- as.character(predic_base$RESULTADO)
predic_base$DADOS <- "Preditos"
train_base$DADOS <- "Atuais"
train_base$RESULTADO <- as.character(train_base$RESULTADO)

base_final <- rbind(predic_base, train_base) %>% as.data.frame()
cum_base <- subset(base_final, base_final$RESULTADO == "confirmado")
cum_base$NUMERO <- 1
cum_base <- cum_base %>%
	group_by(DT_NOT) %>%
	summarise(CASOS = sum(NUMERO, na.rm = T))
cum_base$CUM_CASOS <- cumsum(cum_base$CASOS) 
cum_base$DADOS <- "Preditos"

cum_base <- rbind(cum_train, cum_base) %>% as.data.frame()

ggplot(cum_base, aes(DT_NOT, CUM_CASOS, group = DADOS, color = DADOS))+
	geom_line()+
	theme_bw()+
	labs(y = "Número de Casos", 
	     x = "Data de Notificação")+
	ylim(0,180)+
	theme(axis.text.x = element_text(angle = 90, hjust = 1))




