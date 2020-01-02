library(ggplot2)

simulate_chi <- function(n,figure=TRUE){
  d <- data.frame(a=c(n,n),b=c(n,n))
  nums = c(0)
  p_values  =c(chisq.test(d)$p.value)
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
    fisher = chisq.test(d)$p.value
    if(fisher <= 0.05 & changes_at_significance == -1){
      changes_at_significance = i
    }
    p_values <- append(fisher,p_values)
  }
  p_values <- rev(p_values)
  nums = seq(0,n2)
  
  results = data.frame(Changes=nums,P.Value=p_values)
  
  p <- ggplot(data=results,aes(x=Changes,y=P.Value)) + geom_line() + 
    scale_y_reverse( lim=c(1.00,0.00)) + 
    geom_hline(yintercept=0.05, linetype="dashed", color = "red") + 
    ggtitle(paste('Chi Squared Test (Size = ',(n*4),')'))
  if(figure){
    ggsave(paste('chi_squared_',(n*4),'.png',sep=''),p) 
  }
  return(changes_at_significance)
}

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
    scale_y_reverse( lim=c(1.00,0.00)) + 
    geom_hline(yintercept=0.05, linetype="dashed", color = "red") + 
    ggtitle(paste('Fisher\'s Exact Test (Size = ',(n*4),')'))
  if(figure){
    ggsave(paste('fisherExact_',(n*4),'.png',sep=''),p) 
  }
  return(changes_at_significance)
}

changes_at_sig_Chi = c()
changes_at_sig_Fisher = c()
c = 1
for(i in seq(4,256)){
  c = simulate_chi(i,FALSE)
  f = simulate_fishers(i,FALSE)
  changes_at_sig_Chi = append(c,changes_at_sig_Chi)
  changes_at_sig_Fisher = append(f,changes_at_sig_Fisher)
  print(length(changes_at_sig_Chi))
}
changes_at_sig_Chi <- rev(changes_at_sig_Chi)
changes_at_sig_Fisher <- rev(changes_at_sig_Fisher)
results <- append(changes_at_sig_Chi,changes_at_sig_Fisher)

nums <- (seq(4,256)*4)

chi <- replicate(length(nums),'Chi Squared')
fish <- replicate(length(nums),'Fisher\'s Exact')

type <- append(chi,fish)
nums <- append(nums,nums)
d <- data.frame(Size=nums,Changes=results,Test=type)
d$Ratio <- d$Changes / d$Size

ggplot(data=d,aes(x=Size,y=Changes,color=Test)) + geom_line() + ylab('Changes Required for Significance')+ 
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        axis.title.x=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
ggsave('Analysis-Scripts/Changes-To-Significance/Unused-Figure-1-Chi-Fisher-Changes.png')


ggplot(data=d,aes(x=Size,y=Ratio,color=Test)) + geom_line() + ylab('Ratio: Changes Required for Significance / Size') + 
  theme_bw() +
  theme(plot.title = element_text(size=48), 
        legend.text=element_text(size=20),
        legend.title=element_text(size=20),
        axis.title.y=element_text(size=20),
        axis.title.x=element_text(size=20),
        panel.grid.minor = element_line(size = 1), 
        panel.grid.major = element_line(size = 1),) + 
  scale_fill_manual(values=cbPalette) + 
  scale_colour_manual(values=cbPalette)
ggsave('Analysis-Scripts/Changes-To-Significance/Unused-Figure-2-Chi-Fisher-Ratio.png')

