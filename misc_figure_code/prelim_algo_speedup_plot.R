datas <- data.table("5"=c(8.722372, 5.371310, 1.623881),
                    "10"=c(27.166801, 17.185837, 1.580766),
                    "15"=c(140.110077, 96.897912, 1.445955),
                    "25"=c(453.748933, 312.411563, 1.452407),
                    "label"=c("Serial", "Parallel", "Speedup"))

data.long <- melt(datas, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                  variable.factor=FALSE, variable.name="K Centroids")
data.long$`K Centroids` <- as.numeric(data.long$`K Centroids`)

data.times <- data.long[(label=="Serial" | label=="Parallel")]
names(data.times) <- c("Key", "K Centroids", "Time (S)")

(data.speedup <- data.long[label=="Speedup"])
names(data.speedup)[3] <- "Speedup"
data.speedup

ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Key),
            data=data.times) +
  geom_line(aes(x=`K Centroids`, y=(Speedup-1)*400, color=label), data=data.speedup) +
  scale_y_continuous(sec.axis = sec_axis(trans=~(./400)+1, name="Speedup")) +
  theme(axis.title.y.right = element_text(angle=0, vjust = .5),
        axis.title.y.left = element_text(angle=0, vjust = .5))
