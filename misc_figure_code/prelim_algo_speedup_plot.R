library(data.table)
library(ggplot2)

datas <- data.table("5"=c(8.722372, 3.737203, 2.333930),
                    "10"=c(27.166801, 10.894545, 2.493615),
                    "15"=c(140.110077, 58.985418, 2.375334),
                    "25"=c(453.748933, 208.646101, 2.174729),
                    "label"=c("Serial", "Parallel", "Speedup"))

data.long <- melt(datas, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                  variable.factor=FALSE, variable.name="K Centroids")
data.long$`K Centroids` <- as.numeric(data.long$`K Centroids`)

data.times <- data.long[(label=="Serial" | label=="Parallel")]
names(data.times) <- c("Key", "K Centroids", "Time (S)")

(data.speedup <- data.long[label=="Speedup"])
names(data.speedup)[3] <- "Speedup"

ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Key),
            data=data.times) +
  geom_line(aes(x=`K Centroids`, y=(Speedup-2)*400, color=label), data=data.speedup) +
  scale_y_continuous(sec.axis = sec_axis(trans=~(./400)+2, name="Speedup")) +
  theme(axis.title.y.right = element_text(angle=0, vjust = .5),
        axis.title.y.left = element_text(angle=0, vjust = .5))
