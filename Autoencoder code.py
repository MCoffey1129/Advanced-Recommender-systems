"""
Import the same movielens data
We are going to try encode and decode the movie ratings to try and predict movies which have not been seen
"""

"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import seaborn as sns




"""## Importing the dataset"""

"""## Preparing the training set and the test set"""

"""Import the required datasets
   update the below code to wherever you have saved the datasets"""

movie_lookup = pd.read_csv(r'files\movies.csv')
ratings_train = pd.read_csv(r'files\ratings_train.csv')
ratings_test = pd.read_csv(r'files\ratings_test.csv')
print(movie_lookup)


movie_lookup.shape # 58,098 rows and 3 columns
ratings_train.shape # 198,079 rows and 4 columns
ratings_test.shape # 1,917 rows and 4 columns (we would normally want a larger test dataset approx 25% of the train dataset)
movie_lookup.columns # 3 columns are movieId, title and genre


"""Join the tables tables"""
movie_rec_tn = pd.merge(ratings_train,movie_lookup,how='left', on=['movieId'])
movie_rec_tn.shape # 198,079 rows and 6 columns as expected

movie_rec_test = pd.merge(ratings_test,movie_lookup,how='left', on=['movieId'])
movie_rec_test.shape # 1,917 rows and 6 columns as expected



"""Pivot table with each of the users and movies from the train data"""
movie_rec_tn_pvt = pd.pivot_table(movie_rec_tn, values='rating', index = 'userId', columns='movieId')
movie_rec_tn_pvt.fillna(0,inplace=True)
movie_rec_tn_pvt.head()

zeroised_pvt = pd.DataFrame(0, columns=movie_rec_tn_pvt.columns, index=movie_rec_tn_pvt.index)

movie_rec_test_pvt = pd.pivot_table(movie_rec_test, values='rating', index = 'userId', columns='movieId')
movie_rec_test_pvt.fillna(0,inplace=True)
movie_rec_test_pvt.head()

a = zeroised_pvt.update(movie_rec_test_pvt)
a.head()




"""## Getting the number of users and movies"""
test.shape  # 1012 unique users and 12,699 unique films
nb_users = 1012
nb_movies = 12699



"""## Converting the data into Torch tensors"""
# We are using pytorch - tensors are arrays of elements of a single datatype (multi-dimensional matrix)
# The ratings are going to be the input nodes which go into the neural network
training_tens = torch.FloatTensor(train_np)
test_tens = torch.FloatTensor(test_np)

"""## Creating the architecture of the Neural Network"""
# class can contain variables and functions, preparing a stacked autoencoder
class SAE(nn.Module):
    def __init__(self, ):  # initialise the object of the class
        super(SAE, self).__init__()  # get the classes that are available in nn.Module (we will use nn.Linear)
        self.fc1 = nn.Linear(nb_movies, 20)
        # full connection 1 = nn.Linear(no of movies, no. of neurons in the first hidden layer)
        self.fc2 = nn.Linear(20, 10)  # full connection between the 1st hidden layer and the 2nd hidden layer
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        # the 4th full connection is a connection between the number of neurons in the 3rd layer
        # and the number of movies, remember for autoencoders we are trying to estimate the input layer
        self.activation = nn.Sigmoid()  # got better results using sigmoid rather than the rectifier activation function
    def forward(self, x):
        # The action of encoding and decoding using the sigmoid function
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)  # we do not use the activation for the final layer
        return x
sae = SAE()
criterion = nn.MSELoss()  # criterion for the loss function
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)
# apply stochastic gradient descent to reduce the error, could use the adam optimizer instead but got better results
# with RMSprop
# weight_decay is used to reduce the learning rate after a number of epochs to regulate convergence

"""## Training the SAE"""

nb_epoch = 20  # no. of epochs
for epoch in range(1, nb_epoch + 1):
  train_loss = 0  #  initialise the train loss
  s = 0.  #  no of users who rated at least one movie
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
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
      # backward decides the direction of the correction, optimizer decides the intensity of the correction
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))

"""## Testing the SAE"""
 # think of the test set as the set of movies the user will watch in the future
 # we want to predict what rating we would give that user
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

# You want a test loss of less than 1 star, which means if on average the model will predict a rating of less than 1
# star!!
# For the above we have a test loss of 0.9507, which means that if the model predicts you will give a movie a rating of
# 4 stars you will give it a rating of between 3.05 and 4.95


test_est = []
for i in range(0, nb_movies):
    val = list(output.detach().numpy())[0][i]
    test_est.append(val)

print(len(test_est))


test_ex = pd.DataFrame(test.iloc[1011,:])
print(test_ex)

test_comp = pd.concat([test_ex
              ,pd.DataFrame(test_est, columns=['test_est']).set_index(test_ex.index)],axis=1)



test_comp.columns = ['test_act', 'test_est']

test_comp_red = test_comp.loc[(test_comp['test_act'] != 0) | (test_comp['test_est'] != 0)]

test_comp_titles = pd.merge(test_comp_red, movie_lookup , how='left', left_index = True, right_index=False,
                            right_on='movieId')

test_comp_titles_sort = test_comp_titles.sort_values(by = 'test_act', ascending=False)

test_comp_titles_sort.to_csv(r'Files\test_comp_titles_sort_2.csv', index=True, header=True)