library(data.table)
library(ggplot2)

# TIMING DATA ###########################
lloyd.datas <- data.table("5"=c(3.080444, 1.144337),
                          "10"=c(8.622141, 2.853312),
                          "15"=c(46.224572, 16.703677),
                          "25"=c(154.763342, 53.818381),
                          "label"=c("Optimized Lloyd", "Optimized Lloyd"),
                          "parallelization"=c("Serial", "Parallel"))
lloyd.data.times <- melt(lloyd.datas, id.vars=c("label", "Parallelization"), measure.vars=c("5", "10", "15", "25"),
                  variable.factor=FALSE, variable.name="K Centroids")
lloyd.data.times$`K Centroids` <- as.numeric(lloyd.data.times$`K Centroids`)
names(lloyd.data.times) <- c("Algorithm", "Parallelization", "K Centroids", "Time (S)")
lloyd.data.times

elkan.datas <- data.table("5"=c(4.732397, 2.073992),
                          "10"=c(11.357067, 5.396971),
                          "15"=c(56.625037, 29.008391),
                          "25"=c(181.321405, 119.449632),
                          "label"=c("Elkan", "Elkan"),
                          "parallelization"=c("Serial", "Parallel"))
elkan.data.times <- melt(elkan.datas, id.vars=c("label", "Parallelization"), measure.vars=c("5", "10", "15", "25"),
                        variable.factor=FALSE, variable.name="K Centroids")
elkan.data.times$`K Centroids` <- as.numeric(elkan.data.times$`K Centroids`)
names(elkan.data.times) <- c("Algorithm", "Parallelization", "K Centroids", "Time (S)")
elkan.data.times

slow_datas <- data.table("5"=c(6.41404, 2.65987),
                         "10"=c(18.559297, 7.356104),
                         "15"=c(100.31866, 42.23705),
                         "25"=c(307.145723, 131.894881),
                         "label"=c("Lloyd", "Lloyd"),
                         "parallelization" = c("Serial", "Parallel"))
slow.data.times <- melt(slow_datas, id.vars=c("label", "Parallelization"), measure.vars=c("5", "10", "15", "25"),
                  variable.factor=FALSE, variable.name="K Centroids")
slow.data.times$`K Centroids` <- as.numeric(slow.data.times$`K Centroids`)
names(slow.data.times) <- c("Algorithm", "Parallelization", "K Centroids", "Time (S)")
slow.data.times

# Elkan v. Slow Lloyd plot
slow.times <- rbind(elkan.data.times, slow.data.times)
slow.times
ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Algorithm, linetype=Parallelization), data=slow.times) +
  scale_y_continuous(limits = c(-10, 315)) +
  scale_color_manual(values = c("#FF3300", "#003399"), breaks=c("Elkan", "Lloyd")) +
  guides(color = guide_legend(order=0),
         linetype = guide_legend(order=1)) +
  theme_bw()

# All times plot
all.times <- rbind(slow.times, lloyd.data.times)
ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Algorithm, linetype=Parallelization), data=all.times) +
  scale_y_continuous(limits = c(-10, 315)) +
  guides(color = guide_legend(order=0),
         linetype = guide_legend(order=1)) +
  scale_color_manual(values = c("#FF3300", "#003399", "#66CCCC"),
                     breaks=c("Elkan", "Lloyd", "Optimized Lloyd"),
                     labels = function(x) str_wrap(x, width = 10)) +
  theme_bw()

# SPEEDUP DATA ###########################
slow_speedup <- data.table("5"=c(2.411411084),
                           "10"=c(2.522979148),
                           "15"=c(2.375134154),
                           "25"=c(2.328716025),
                           "label"=c("Lloyd"))
slow.data.speedup <- melt(slow_speedup, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                     variable.factor=FALSE, variable.name="K Centroids")
slow.data.speedup$`K Centroids` <- as.numeric(slow.data.speedup$`K Centroids`)
names(slow.data.speedup) <- c("Algorithm", "K Centroids", "Time (S)")
slow.data.speedup

speedup_datas <- data.table("5"=c(2.691902822, 2.281782),
                            "10"=c(3.021800981, 2.104341),
                            "15"=c(2.767329134, 1.952023),
                            "25"=c(2.875659563, 1.517973743),
                            "label"=c("Optimized Lloyd", "Elkan"))
data.speedup <- melt(speedup_datas, id.vars="label", measure.vars=c("5", "10", "15", "25"),
                     variable.factor=FALSE, variable.name="K Centroids")
data.speedup$`K Centroids` <- as.numeric(data.speedup$`K Centroids`)
names(data.speedup) <- c("Algorithm", "K Centroids", "Time (S)")

data.speedup <- rbind(slow.data.speedup, data.speedup)
data.speedup

# Speedup Plot
ggplot() +
  geom_line(aes(x=`K Centroids`, y=`Time (S)`, color=Algorithm), data=data.speedup) +
  scale_y_continuous(limits = c(0, 4.25)) +
  theme_bw()
