library(xgboost)
library(nnet)
library(stargazer)
library(ggplot2)
library(readr)
library(gtable)
library(grid)
library(BBmisc)

vColors = c("#1269B0","#72791C","#91056A")

# Run script to produce importance plot and multinominal regression output.
# Images are saved in /image folder

#### Data import and transformation ####

dSprites_LSF = read.csv("./data/dSprites/dSprites_LSF_samples.csv")
dSprites_LSF$Label = factor(as.character(dSprites_LSF$Label), labels = c("square","ellipse","heart"))

dSprites_LSF$Label = relevel(dSprites_LSF$Label, ref = "ellipse")

dSprites_LSF[,2:11] = normalize(dSprites_LSF[,2:11])

#### Multinominal regression for disnt. metric ####

model <- multinom(Label ~., data=dSprites_LSF[,-c(3,6)]) # cbind(dSprites_LSF[,-c(3,6)], dSprites_LSF$LSF.0^2)

print("Multinominal model with square as baseline")
summary(model)

# stargazer(model) # for LaTeX output

z <- summary(model)$coefficients/summary(model)$standard.errors
print("Standardized coefficients")
z

print("2-tailed z test")
p <- (1 - pnorm(abs(z), 0, 1)) * 2
p

pp <- fitted(model)
data.frame(pp[5:10,], True = dSprites_LSF$Label[5:10])

# Logistic Regression

model <- glm(Label ~.,family=binomial(link='logit'),data=dSprites_LSF[dSprites_LSF$Label != "heart",-c(3,6)])
model <- glm(Label ~.,family=binomial(link='logit'),data=dSprites_LSF[dSprites_LSF$Label != "square",-c(3,6)])

# stargazer(model) # for LaTeX output

summary(model)

#### Feature Importance ####

dSprites_LSF$Label = relevel(dSprites_LSF$Label, ref = "square")

model_boost <- xgboost(data = as.matrix(dSprites_LSF[,-c(1)]), 
                    label = as.integer(dSprites_LSF$Label)-1, 
                    max.depth = 4, 
                    eta = 1, 
                    nthread = 2, 
                    nrounds = 200,
                    num_class = 3,
                    objective = "multi:softmax",
                    eval_metric = "mlogloss",
                    verbose= 0
                )


plot0 =
xgb.ggplot.importance(xgb.importance(model = model_boost), n_clusters = 3) + 
  theme_minimal() + 
  coord_cartesian() + 
  scale_fill_manual(values = vColors[c(2,3,1)]) + 
  theme(axis.text.x = element_text(angle = 0, hjust = 1)) + 
  ggtitle(label="") +
  xlab("Latent Space Features")

#### Effect of Beta ####

dfData <- data.frame(loss = c(67.8,36.9,12.8,-17.5,-48.7,-83.6,-121,-157,-191,-230),
                    accuracy = c(45.61,51.09,62.48,58.38,62.41,63.63,56.79,56.83,55.78,50.38)/100
                )

plot1 =
ggplot(data = dfData, aes(x=1:10, y=accuracy)) + 
  geom_line(cex = 1, color = vColors[1]) +
  geom_point(cex = 2, color = vColors[1]) +
  ylab("Test Accuracy") +
  coord_cartesian(ylim = c(0.4, 1)) +
  scale_x_continuous(expression(beta), labels = as.character(1:10), breaks = 1:10) +
  theme(panel.grid.minor.x = element_blank()) +
  theme(axis.title.x=element_blank(), axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
  scale_y_continuous(labels = scales::percent)

plot2 =
ggplot(data = dfData, aes(x=1:10, y=loss)) + 
  geom_line(cex = 1, color = vColors[2]) +
  geom_point(cex = 2, color = vColors[2]) +
  ylab("\u03B2-TCVAE loss") +
  scale_x_continuous(expression(beta), labels = as.character(1:10), breaks = 1:10) +
  theme(panel.grid.minor.x = element_blank()) 


g1 <- ggplotGrob(plot0)
g2 <- ggplotGrob(plot1)
g3 <- ggplotGrob(plot2)
g <- rbind(g2, g3, size = "first")
g$widths <- unit.pmax(g2$widths, g3$widths)

grid =
grid.arrange(g, g1, ncol=2)

ggsave("./images/dis_metric.pdf", grid, width = 12, height = 5, device='pdf', dpi=400)