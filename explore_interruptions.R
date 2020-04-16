data <- read.csv('/Users/samanthagoerger/Desktop/Thesis/Data/gender.csv')
utterances_og <- read.csv('/Users/samanthagoerger/Desktop/Thesis/Data/data_full.csv')
utterances <- utterances_og[utterances_og$gender != "J",]
length(utterances$gender[utterances$gender == "M"])
library(dplyr)
library(stringr)

sum(utterances_mf$interrupt)
1 - mean(utterances_mf$interrupt)
####################################

sum(data$speaker == "GENERAL VERRILLI:")

general_cases <- data$case_id[data$speaker == "GENERAL VERRILLI:"]

general <- data[data$case_id %in% general_cases,]

n_speakers <- general %>%
  group_by(case_id)%>%
  summarize(n = n())
table(n_speakers$n)

order(unique(general$case_id))

male <- general %>%
  filter(gender == 'M') %>%
  group_by(speaker) %>%
  summarize(n = n())

#length(female$case_id)
#fem_cases <- unique(female$case_id)

#n_speakers_fem <- general %>%
#  filter(case_id %in% fem_cases) %>%
#  group_by(case_id) %>%
#  summarize(n = n())
#table(n_speakers_fem$n)
 
#write.csv(general, '/Users/samanthagoerger/Desktop/Thesis/Data/sg_cases.csv')

utterances$year <- substr(as.character(utterances$case_id), 1, 4)



table(utterances$gender)


utterances$interrupt_new <- NA
utterances$text_new <- str_replace_all(utterances$text, '\n', '')
utterances$text_new2 <- str_replace_all(utterances$text_new, '-.', '--')

utterances$interrupt_new <- ifelse(endsWith(utterances$text_new2, '-'), 1, 0)

sum(utterances$interrupt)
sum(utterances$interrupt_new)  

t.test(utterances$interrupt_new[utterances$gender == 'F' & utterances$year == '2015'],
       utterances$interrupt_new[utterances$gender == 'M' & utterances$year == '2015'])
ttest_all <- t.test(utterances$interrupt[utterances$gender == 'F'],
       utterances$interrupt[utterances$gender == 'M'])

utter_general <- utterances[utterances$case_id %in% general_cases & utterances$gender != 'J',]
utter_general_n <- utter_general[utter_general$speaker != 'GENERAL VERRILLI:',]

t.test(utter_general$interrupt_new[utter_general$gender == 'F' & utter_general$year == '2010'],
       utter_general$interrupt_new[utter_general$gender == 'M' & utter_general$year == '2010'])
ttest_sg <- t.test(utter_general$interrupt[utter_general$gender == 'F'],
       utter_general$interrupt[utter_general$gender == 'M'])
mean(utter_general$interrupt_new[utter_general$gender == 'F'])

utter2015 <- utterances[utterances$year == '2015',]
  
length(data$speaker[data$speaker == "MR. CLEMENT:"])

n_speakers <- data %>%
  group_by(speaker)%>%
  summarize(n = n())

sum(n_speakers$n[n_speakers$n >= 5])

mean(utterances$interrupt == 0)

1 - mean(utterances$interrupt[utterances$gender == "M"])

general_wogeneral <- general[general$speaker != 'GENERAL VERRILLI:',]
table(general_wogeneral$gender)
sum(general_wogeneral$gender == "M")/nrow(general_wogeneral)

mf <- data[data$gender != "J",]
sum(mf$gender == "M")/nrow(mf)

utter_general_mf <- utter_general[utter_general$gender != "J",]
table(utter_general_mf$gender)

sum(general$gender == "M")/nrow(general)

full_diff <- ttest_all$estimate[1] - ttest_all$estimate[2]
ttest_all$conf.int

sg_diff <- ttest_sg$estimate[1] - ttest_sg$estimate[2]
ttest_sg$conf.int

sg_x <- 1.5
plot(50, 50, type = "n",
     xlim = c(0.9,sg_x + .2),
     ylim = c(-0.02,0.07),
     xlab = "",
     ylab = "Interruption rates (D. in Means)",
     main = "Difference in interruptions by gender",
     cex.lab = 0.75,
     xaxt = "n")
axis(side = 1, at = c(1.1, sg_x), labels = c("Full Data", "Solicitor General"))
abline(h = 0)

points(x = c(1.1), y = c(full_diff), pch = c(18), 
       cex = 1.5, col = c("darkred"))
points(x = c(sg_x), y = c(sg_diff), pch = c(18), 
       cex = 1.5, col = c("darkred"))
  
lines(x = c(1.1,1.1), y = ttest_all$conf.int, col = "darkred", lwd = 2)
lines(x = c(sg_x,sg_x), y = ttest_sg$conf.int, col = "darkred", lwd = 2)

