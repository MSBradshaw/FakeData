library(ggplot2)

simulate_fishers <- function(n,figure=TRUE){
  d <- data.frame(a=c(n,n),b=c(n,n))
  nums = c(0)
  p_values  =c(fisher.test(d)$p.value)
  n2 <- n * 2
  changes_at_significance = -1
  for( i in seq(1,n2)){
    #move 1
    if( i %% 2 == 0){
      #move false negative to true positive
      d[1,1] = d[1,1] + 1
      d[1,2] = d[1,2] - 1
    }else{
      #move false postiive to true negative
      d[2,2] = d[2,2] + 1
      d[2,1] = d[2,1] - 1
    }
    fisher = fisher.test(d)$p.value
    if(fisher <= 0.05 & changes_at_significance == -1){
      changes_at_significance = i
    }
    p_values <- append(fisher,p_values)
  }
  p_values <- rev(p_values)
  nums = seq(0,n2)
  
  results = data.frame(Changes=nums,P.Value=p_values)
  
  p <- ggplot(data=results,aes(x=Changes,y=P.Value)) + geom_line() + 
    ylab('p-value') + 
    xlab('Number of Changes') +
    scale_y_reverse( lim=c(1.00,0.00)) + 
    geom_hline(yintercept=0.05, linetype="dashed", color = "red") + 
    ggtitle(paste('Fisher\'s Exact Test (Size = ',(n*4),')'))  + 
    theme_bw() +
    theme(panel.grid.minor = element_line(size = 1), 
          panel.grid.major = element_line(size = 1),) + 
    scale_fill_manual(values=cbPalette) + 
    scale_colour_manual(values=cbPalette)
  if(figure){
    ggsave(paste('Analysis-Scripts/Changes-To-Significance/fisherExact_',(n*4),'.png',sep=''),p) 
  }
  return(p)
}

p4 <- simulate_fishers(4,FALSE)
p8 <- simulate_fishers(8,FALSE)
p16 <- simulate_fishers(16,FALSE)
p32 <- simulate_fishers(32,FALSE)
p64 <- simulate_fishers(64,FALSE)
p128 <- simulate_fishers(128,FALSE)

ggarrange(p4,p8,p16,p32,p64,p128,ncol=2,nrow=3)
ggsave('Analysis-Scripts/Changes-To-Significance/Unused-Figure-3-Fishers-Pannel.png',units='in',width = 8.5, height = 11) 















