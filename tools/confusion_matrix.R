library(heatmaply)
library(RColorBrewer)

# Run script to produce confusion matrix plots. Data is already in the script.
# Images are saved in /image folder

# MNIST
cm_mnist = 
as.integer(c(
910,    0,   15,    8,    0,   30,   13,    2,    2,    0,
0, 1115,    7,    2,    1,    0,    5,    1,    4,    0,
16,    4,  963,    8,    0,    0,    7,   19,   14,    1,
0,    1,   14,  948,    0,   25,    0,   14,    6,    2,
0,    0,    2,    0,  939,    0,   14,   14,    0,   13,
9,    0,    7,   52,    5,  785,   18,    4,   10,    2,
16,    2,   14,    1,    3,   27,  895,    0,    0,    0,
1,    6,   27,    2,    8,    0,    0,  942,    0,   42,
21,    2,   24,   36,    4,   60,   11,    8,  801,    7,
6,    7,    7,   10,   37,    5,    1,   47,   16,  873))


cm_mnist = as.data.frame(matrix(cm_mnist, ncol=10, byrow=TRUE))

colnames(cm_mnist) = 0:9
rownames(cm_mnist) = 0:9

fig = 
heatmaply(
  cm_mnist,
  file= "./images/conf_mat_mnist.png",
  key = FALSE,
  colors = colorRampPalette(c("#FFFFFF","#007A96", "#91056A")) (4),
  dendrogram = "none",
  grid_color = "white",
  plot_method = "plotly",
  cellnote = cm_mnist,
  xlab = "Predicted",
  ylab = "True",
  Colv = TRUE,
  Rowv = TRUE,
  width = 1200,
  height = 600,
  column_text_angle = 0,
  hide_colorbar = TRUE,
  fontsize_row = 14,
  fontsize_col = 14,
  cellnote_size = 16,
  cellnote_textposition = "middle center"
)

# dSprites
cm_dSprites = 
as.integer(c(
14674,  1070,  5988,
4020, 12368,  5098,
5231,  2156, 14395))


cm_dSprites = as.data.frame(matrix(cm_dSprites, ncol=3, byrow=TRUE))

colnames(cm_dSprites) = c("square", "ellipse", "heart")
rownames(cm_dSprites) = c("square", "ellipse", "heart")

fig = 
heatmaply(
  cm_dSprites,
  file= "./images/conf_mat_dSprites.png",
  key = FALSE,
  colors = colorRampPalette(c("#FFFFFF","#007A96", "#91056A")) (4),
  dendrogram = "none",
  grid_color = "white",
  plot_method = "plotly",
  cellnote = cm_dSprites,
  xlab = "Predicted",
  ylab = "True",
  Colv = TRUE,
  Rowv = TRUE,
  width = 800,
  height = 600,
  column_text_angle = 0,
  hide_colorbar = TRUE,
  fontsize_row = 20,
  fontsize_col = 20,
  cellnote_size = 24,
  cellnote_textposition = "middle center"
)

# OCT retina
cm_OCT = 
as.integer(c(
141,  96,   7,   6,
19, 220,   4,   7,
68,  51,  47,  84,
26,  45,  34, 145))


cm_OCT = as.data.frame(matrix(cm_OCT, ncol=4, byrow=TRUE))

colnames(cm_OCT) = c("CNV", "DME", "Drusen","Normal")
rownames(cm_OCT) = c("CNV", "DME", "Drusen","Normal")

fig = 
heatmaply(
  cm_OCT,
  file= "./images/conf_mat_OCT.png",
  key = FALSE,
  colors = colorRampPalette(c("#FFFFFF","#007A96", "#91056A")) (4),
  dendrogram = "none",
  grid_color = "white",
  plot_method = "plotly",
  cellnote = cm_OCT,
  xlab = "Predicted",
  ylab = "True",
  Colv = TRUE,
  Rowv = TRUE,
  width = 800,
  height = 600,
  column_text_angle = 0,
  hide_colorbar = TRUE,
  fontsize_row = 20,
  fontsize_col = 20,
  cellnote_size = 26,
  cellnote_textposition = "middle center"
)
