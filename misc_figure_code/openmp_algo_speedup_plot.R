library(data.table)
library(ggplot2)

datas <- data.table("5"=c(3.080444, 1.144337, 4.732397, 2.073992),
                    "10"=c(8.622141, 2.853312, 11.357067, 5.396971),
                    "15"=c(46.224572, 16.703677, 56.625037, 29.008391),
                    "25"=c(154.763342, 53.818381, 181.321405, 119.449632),
                    "label"=c("Lloyd Serial", "Lloyd Parallel", "Elkan Serial", "Elkan Parallel"))

speedup_datas <- data.table("5"=c(2.691902822, 2.281782),
                            "10"=c(3.021800981, 2.104341),
                            "15"=c(2.767329134, 1.952023),
                            "25"=c(2.875659563, 1.517973743),
                            "label"=c("Lloyd's", "Elkan"))


data.times <- melt(datas, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                  variable.factor=FALSE, variable.name="K Centroids")
data.times$`K Centroids` <- as.numeric(data.times$`K Centroids`)
names(data.times) <- c("Algorithm", "K Centroids", "Time (S)")

data.speedup <- melt(speedup_datas, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                     variable.factor=FALSE, variable.name="K Centroids")
data.speedup$`K Centroids` <- as.numeric(data.speedup$`K Centroids`)
names(data.speedup) <- c("Algorithm", "K Centroids", "Time (S)")

ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Algorithm), data=data.times)

ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Algorithm), data=data.speedup) +
  scale_y_continuous(limits = c(0, 3.25))
