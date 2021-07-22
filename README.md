Deep AutoEncoders (Pytorch) - as a method for predicting movie ratings
===


The code attached uses Deep Autoencoders to predict movie ratings in the future.  
The Training dataset contains ratings users have given movies to date (assumption).  
The Test dataset contains movie ratings the users will give in the future (assumption). The model will be trained to predict these Test values.    
The dataset used is a subset of the famous MovieLens dataset.  

Please note the model has not been run using K-fold cross validation as we are working with small datasets but
I would advice doing that going forward, I would also use a larger Test dataset.


Results
---

Test loss of 0.8930, which means that the model on average is within 1 star of the actual movie ratings for users in the Test set.
Please note we would need a larger Training sample to get a strong recommender model


Example
---
Looking at the last entry in the Test dataset, user 2025 gave movie 160 (Congo [1995] which is an "Action|Advernture|Mystery|Sci-fi") a rating of 2. 
The model is forecasting that the user will rate the movie a value of 2.6 which at a high level means that the model forecasts that the user will not like
this movie which is a successful prediction.


Any questions, please ask.
