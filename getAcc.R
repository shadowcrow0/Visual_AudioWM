dat <- read.csv("clean_yyc.csv")
dat <-subset(dat, trial !="P")


dat<- dat_niki
#colSums(is.na(dat))
dat <- dat[, c("trial", "RT","Response","Probe","condition","Acc")]
#"Channel1","Channel2",
dat <- dat[!is.na(dat$trial), ]
dat<- dat[-(1:50),]
dat <- dat[!is.na(dat$Probe), ]
# Condition:
dat$Channel1 <- NA
dat$Channel2 <- NA

dat$Channel1[dat$condition ==0 | dat$condition ==1] <- 0
dat$Channel2[dat$condition ==0 | dat$condition ==1] <- 0

dat$Channel1[dat$condition ==2 | dat$condition ==3] <- 2
dat$Channel2[dat$condition ==2 | dat$condition ==3] <- 0

dat$Channel1[dat$condition ==4 | dat$condition ==5] <- 1
dat$Channel2[dat$condition ==4 | dat$condition ==5] <- 0

dat$Channel1[dat$condition ==6 | dat$condition ==7] <- 0
dat$Channel2[dat$condition ==6 | dat$condition ==7] <- 2

dat$Channel1[dat$condition ==8 | dat$condition ==9] <- 2
dat$Channel2[dat$condition ==8 | dat$condition ==9] <- 2

dat$Channel1[dat$condition ==10 | dat$condition ==11] <- 1
dat$Channel2[dat$condition ==10 | dat$condition ==11] <- 2

dat$Channel1[dat$condition ==12 | dat$condition ==13] <- 0
dat$Channel2[dat$condition ==12 | dat$condition ==13] <- 1

dat$Channel1[dat$condition ==14 | dat$condition ==15] <- 2
dat$Channel2[dat$condition ==14 | dat$condition ==15] <- 1

dat$Channel1[dat$condition ==16 | dat$condition ==17] <- 1
dat$Channel2[dat$condition ==16 | dat$condition ==17] <- 1
dat$Acc <- NA
dat$Acc[dat$Response == 4 & dat$condition ==0] =1
dat$Acc[dat$Response == 4 & dat$condition ==1] =1
dat$Acc[dat$Response != 4 & dat$condition ==0] =0
dat$Acc[dat$Response != 4 & dat$condition ==1] =0
dat$Acc[dat$Response == 4 & dat$condition ==2] =0
dat$Acc[dat$Response == 4 & dat$condition ==3] =0
dat$Acc[dat$Response == 3 & dat$condition ==2] =1
dat$Acc[dat$Response == 3 & dat$condition ==3] =1
dat$Acc[dat$Response == 3 & dat$condition ==4] =1
dat$Acc[dat$Response == 3 & dat$condition ==5] =1
dat$Acc[dat$Response == 4 & dat$condition ==4] =0
dat$Acc[dat$Response == 4 & dat$condition ==5] =0
dat$Acc[dat$Response == 3 & dat$condition ==6] =1
dat$Acc[dat$Response == 3 & dat$condition ==7] =1
dat$Acc[dat$Response == 4 & dat$condition ==6] =0
dat$Acc[dat$Response == 4 & dat$condition ==7] =0
dat$Acc[dat$Response == 3 & dat$condition ==8] =1
dat$Acc[dat$Response == 3 & dat$condition ==9] =1
dat$Acc[dat$Response == 4 & dat$condition ==8] =0
dat$Acc[dat$Response == 4 & dat$condition ==9] =0
dat$Acc[dat$Response == 3 & dat$condition ==10] =1
dat$Acc[dat$Response == 3 & dat$condition ==11] =1
dat$Acc[dat$Response == 4 & dat$condition ==10] =0
dat$Acc[dat$Response == 4 & dat$condition ==11] =0
dat$Acc[dat$Response == 3 & dat$condition ==12] =1
dat$Acc[dat$Response == 3 & dat$condition ==13] =1
dat$Acc[dat$Response == 4 & dat$condition ==12] =0
dat$Acc[dat$Response == 4 & dat$condition ==13] =0
dat$Acc[dat$Response == 3 & dat$condition ==14] =1
dat$Acc[dat$Response == 3 & dat$condition ==15] =1
dat$Acc[dat$Response == 4 & dat$condition ==14] =0
dat$Acc[dat$Response == 4 & dat$condition ==15] =0
dat$Acc[dat$Response == 3 & dat$condition ==16] =1
dat$Acc[dat$Response == 3 & dat$condition ==17] =1
dat$Acc[dat$Response == 4 & dat$condition ==16] =0
dat$Acc[dat$Response == 4 & dat$condition ==17] =0
dat$Channel1 <- factor(dat$Channel1)
dat$Channel2 <- factor(dat$Channel2)
dat <- dat[!is.na(dat$Channel1), ]
dat <- dat[!is.na(dat$RT), ]
summary(dat)

# 看有幾筆符合條件
sum(dat$Channel1 == 1 & dat$Channel2 == 1, na.rm = TRUE)



with(subset(dat, Acc==1), plot(ecdf(RT[Channel1==1 & Channel2==1]), col='blue'))
with(subset(dat, Acc==1), lines(ecdf(RT[Channel1==2 & Channel2==1]), col='purple'))
with(subset(dat, Acc==1), lines(ecdf(RT[Channel1==1 & Channel2==2]), col='orange'))
with(subset(dat, Acc==1), lines(ecdf(RT[Channel1==2 & Channel2==2]), col='red'))

with(subset(dat, Acc==1), plot(density(RT[Channel1==1 & Channel2==1]), col='blue'))
with(subset(dat, Acc==1), lines(density(RT[Channel1==2 & Channel2==1]), col='purple'))
with(subset(dat, Acc==1), lines(density(RT[Channel1==1 & Channel2==2]), col='orange'))
with(subset(dat, Acc==1), lines(density(RT[Channel1==2 & Channel2==2]), col='red'))

with(subset(dat, Acc==1), plot(ecdf(RT[Channel1 > 0 & Channel2==0]), col='blue'))
with(subset(dat, Acc==1), lines(ecdf(RT[Channel1==0 & Channel2> 1]), col='purple'))
with(subset(dat, Acc==1), lines(ecdf(RT[Channel1>0 & Channel2> 1]), col='red'))


#tapply(df$value, df$factor, mean)

dat$Probe <- factor(dat$Probe)
tapply(dat$RT, dat$Probe, mean)
tapply(dat$Acc, dat$Probe, mean)
