#################################################################################################################
        # Deep AutoEncoders (Pytorch) - as a method for predicting movie ratings
#################################################################################################################

# We will use Deep Autoencoders to predict ratings users will give movies in the future.
# We will assume that the Training dataset is the ratings users have given movies to date and that
# the Test dataset contains movie ratings the users will give in the future.
# The dataset used is a subset of the famous MovieLens dataset.
# Please note the model has not been run using K-fold cross validation as we are working with small datasets but
# I would advice doing that going forward, I would also use a larger Test dataset.

"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable


"""Import the required datasets
   update the below code to wherever you have saved the datasets"""

movie_lookup = pd.read_csv(r'files\movies.csv')
ratings_train = pd.read_csv(r'files\ratings_train.csv')
ratings_test = pd.read_csv(r'files\ratings_test.csv')
print(movie_lookup)


movie_lookup.shape # 58,098 rows and 3 columns
movie_lookup.columns # 3 columns are movieId, title and genre
ratings_train.shape # 198,118 rows and 4 columns
ratings_test.shape # 1,878 rows and 4 columns (we would normally want a larger test dataset approx 25% of the train dataset)



"""Join the tables with the train and test tables with the lookup data"""
movie_rec_tn = pd.merge(ratings_train,movie_lookup,how='left', on=['movieId'])
movie_rec_tn.shape # 198,118 rows and 6 columns as expected

movie_rec_test = pd.merge(ratings_test,movie_lookup,how='left', on=['movieId'])
movie_rec_test.shape # 1,878 rows and 6 columns as expected



"""Pivot table with each of the users and movies from the train data"""
movie_rec_tn_pvt = pd.pivot_table(movie_rec_tn, values='rating', index = 'userId', columns='movieId')
movie_rec_tn_pvt.fillna(0,inplace=True)
movie_rec_tn_pvt.head()

"""We need to get the Test dataset in the exact same shape as the training data"""
zeroised_pvt = pd.DataFrame(np.NAN, columns=movie_rec_tn_pvt.columns, index=movie_rec_tn_pvt.index)

movie_rec_test_pvt = pd.pivot_table(movie_rec_test, values='rating', index = 'userId', columns='movieId')
movie_rec_test_pvt.fillna(0,inplace=True)
movie_rec_test_pvt.head()

movie_rec_test_pvt_fn = zeroised_pvt.combine_first(movie_rec_test_pvt)
movie_rec_test_pvt_fn.fillna(0,inplace=True)
movie_rec_test_pvt_fn.iloc[27,:].describe()   # is 5
movie_rec_test_pvt_fn.tail()


"""## Getting the number of users and movies"""
movie_rec_tn_pvt.shape  # 2025 unique users and 12,699 unique movies
nb_users = 2025
nb_movies = 12699


"""# Create a train and test numpy array required for pytorch"""
train_np = movie_rec_tn_pvt.values
test_np = movie_rec_test_pvt_fn.values


"""## Converting the data into Torch tensors"""
# We are using pytorch - tensors are arrays of elements of a single datatype (multi-dimensional matrix)
# The ratings are going to be the input nodes which go into the neural network
training_tens = torch.FloatTensor(train_np)
test_tens = torch.FloatTensor(test_np)

"""manual_seed is used in order to ensure reproducibility """
torch.manual_seed(0)

"""## Creating the architecture of the Neural Network"""
# class can contain variables and functions we will prepare a stacked Autoencoder below
class SAE(nn.Module):
    def __init__(self, ):  # initialise the object of the class
        super(SAE, self).__init__()  # get the classes that are available in nn.Module (we will use nn.Linear)
        self.fc1 = nn.Linear(nb_movies, 25)
        # full connection 1 = nn.Linear(no of movies, no. of neurons in the first hidden layer)
        self.fc2 = nn.Linear(25, 10)  # full connection between the 1st hidden layer and the 2nd hidden layer
        self.fc3 = nn.Linear(10, 25)
        self.fc4 = nn.Linear(25, nb_movies)
        # the 4th full connection is a connection between the number of neurons in the 3rd layer
        # and the number of movies, remember for autoencoders we are trying to estimate the input layer
        self.activation = nn.Sigmoid()  # we use sigmoid as the activation function
        # self.activation = nn.ReLU()  # see if you get better results using the rectified linear unit
    def forward(self, x):
        # The action of encoding and decoding using the sigmoid function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # we do not use the activation function for the final layer
        return x
sae = SAE()
# criterion for the loss function, we use mean squared error if the output was categorical I would use CrossEntropy
criterion = nn.MSELoss()

# Apply stochastic gradient descent to reduce the error, the learning rate used may be a little too high
# weight_decay is used to reduce the learning rate after a number of epochs to regulate convergence
# you can also try optim.Adam to see if you get better results
optimizer = optim.RMSprop(sae.parameters(), lr = 0.02, weight_decay = 0.4)


"""## Training the SAE"""

nb_epoch = 100  # no. of epochs
for epoch in range(1, nb_epoch + 1):
  train_loss = 0  #  initialise the train loss
  s = 0.  #  users who rated at least one movie
  for id_user in range(nb_users):
    input = Variable(training_tens[id_user]).unsqueeze(0)
    # input vector of feature of this particular user, we need to create a batch in order for pytorch to accept the data
    # thus is the reason why we use Variable().unsqueeze()
    target = input.clone()  # target is equal to the input
    if torch.sum(target.data > 0) > 0:
      # if condition is used to optimize the memory (we only want to look at users who rated at least one movie)
      output = sae(input)  # vector of predicted ratings
      target.require_grad = False  # ensure that we do not compute the gradient with respect to the target
      output[target == 0] = 0  # these values will not impact the error
      loss = criterion(output, target)  # loss is equal to the mse between the output and the target
      mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)  # only non zero movies
      # mean_corrector represents the average of the error but only for movies which were rated
      loss.backward()  # do we need to increase or decrease the weights.
      train_loss += np.sqrt(loss.data*mean_corrector)  # train loss
      s += 1.
      optimizer.step()
      # backward decides the direction of the correction, optimizer decides the intensity of the correction
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

"""## Testing the SAE"""
# Think of the test set as the set of movies the user will watch in the future we want to predict what movies
# we would recommend to that user by predicting a rating for the movies in the Test set
test_loss = 0
s = 0.
for id_user in range(nb_users):
  input = Variable(training_tens[id_user]).unsqueeze(0)  # input is the training data
  target = Variable(test_tens[id_user]).unsqueeze(0)  # target is the test data
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_movies/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))


# For the above we have a test loss of 0.8930, which means that the model on average is within 1 star
# of the actual movie ratings for users in the Test set. Please note we would need a larger Training sample
# to get a strong recommender model


#################################################################################################################
# Example - Looking at the last entry in the Test dataset, user 2025 gave movie 160 (Congo [1995]
# which is an "Action|Advernture|Mystery|Sci-fi") a rating of 2, we would expect the model to come up with
# a low rating for this movie and as such not recommend the movie to the user (otherwise our recommender model
# is not doing its job!!!)
#################################################################################################################

# The forecasted rating user 2025 will give the movie Congo [1995]
# .detach().numpy() is used to transform a pytorch value into an array
test_est = []
for i in range(0, nb_movies):
    val = list(output.detach().numpy())[0][i]
    test_est.append(val)

test_ex = pd.DataFrame(movie_rec_test_pvt_fn.iloc[nb_users-1,:])
print(test_ex)


# create a dataframe of forecasted v actual
test_comp = pd.concat([test_ex
              ,pd.DataFrame(test_est, columns=['test_est']).set_index(test_ex.index)],axis=1)

test_comp.columns = ['test_act', 'test_est']

# Only keep non-zero values
test_comp_red = test_comp.loc[(test_comp['test_act'] != 0) | (test_comp['test_est'] != 0)]

# Join on the movie name and genre
test_comp_titles = pd.merge(test_comp_red, movie_lookup , how='left', left_index = True, right_index=False,
                            right_on='movieId')

# Write out results to a csv - this fail for you if you do not have a folder called Files
test_comp_titles.to_csv(r'Files\test_comp_titles.csv', index=True, header=True)

#################################################################################################################
# Result - the model is forecasting that the user will rate the movie a value of 2.6
# which at a high level means that the model forecasts that the user will not like
# this movie which is a successful prediction given that the user gave the movie a rating of 2.
#################################################################################################################