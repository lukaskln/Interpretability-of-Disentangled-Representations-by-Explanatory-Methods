#### Setup ####

library(ggplot2)
library(readr)
library(gtable)
library(grid)

vColors = c("#1269B0","#72791C","#91056A")

theme_set(theme_minimal())

#### Effect of beta ####

dfData <- data.frame(MNIST = c(94.80,93.24,	93.46,	91.98,	92.54,	92.83,	91.76,	91.43,	90.59,	87.56)/100,
                    dSprites = c(96.39,	96.70,	96.44,	91.93,	85.64,	87.06,	76.33,	73.87,	72.12,	72.16)/100,
                    OCT = c(48.80,	45.70,	48.20,	43.60,	49.20,	49.40,	48.80,	40.00,	46.90,	47.40)/100
                )


dfData <- reshape2::melt(dfData)

ggplot(data = dfData, aes(x=rep(1:10,3), y=value)) + 
geom_line(aes(colour=variable), cex = 1) +
geom_point(aes(colour=variable), cex = 2) +
scale_fill_manual(values = vColors) +
scale_color_manual(values = vColors) + 
ylab("Test Accuracy") +
scale_x_continuous(expression(beta), labels = as.character(1:10), breaks = 1:10) +
theme(panel.grid.minor.x = element_blank()) + 
guides(colour=guide_legend(title="Dataset")) +
coord_cartesian(ylim = c(0.4, 1)) +
scale_y_continuous(labels = scales::percent)

ggsave("./images/beta_effect.pdf", width = 9, height = 5, device='pdf', dpi=300)

