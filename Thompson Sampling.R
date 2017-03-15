#Upper Confidence bound

#Data Peocessing
setwd("C://Vivek//Learnings//Machine Learning Udemy")

#Read Data Set
dataset = read.csv("Ads_CTR_Optimisation.csv")

#Implementing UCB
N = 10000
d = 10
ads_selected = integer()
numbers_of_reward_1 = integer(d)
numbers_of_reward_0 = integer(d)
total_rewards = 0
for (n in 1:N){
  ad = 0
  max_random = 0
  for (i in 1:d){
    random_beta = rbeta(n = 1, shape1 = numbers_of_reward_1[i] + 1, shape2 = numbers_of_reward_0[i] + 1 )
    if(random_beta > max_random){
      max_random = random_beta
      ad = i
    }
    
  }
  ads_selected = append(ads_selected,ad)
  reward = dataset[n, ad]
  if(reward == 1){
    numbers_of_reward_1[ad] = numbers_of_reward_1[ad] + 1  
  }else{
    numbers_of_reward_0[ad] = numbers_of_reward_0[ad] + 1
  }
  total_rewards = total_rewards + reward
  
  
}

#Vizualizing
hist(ads_selected, col='blue')
