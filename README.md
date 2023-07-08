# Predicting the Winner of Horse Racing
The government in Hong Kong administrates one of the most popular sports in the city, Horse Racing.  The results, as well as the accompaning information that bettors use to inform their decisions are all publically availible.  

## The process:
To engineer the features for this prediction, I aggregated each horses performance over the past 5 races, as well as creating statistics to quantify how other horses from the same sire and same trainer and same jockey had performed.  These statistics were then used to fit a logistic regression model which was then used to predict winners of races on unseen data.  This model's preformance on unseen data beat the public model by several percentage points.  
