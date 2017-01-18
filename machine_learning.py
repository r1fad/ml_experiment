'''
The dataset contains information about video games
I use the critic score and North American sales and European Sales to predict Global Sales
'''

#Import required library
import pandas
import matplotlib.pyplot as plt
import pickle
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Read in the data.
video_games = pandas.read_csv("video_games.csv")

# Remove any rows without gross
video_games = video_games[video_games["Global_Sales"] > 0]

# Remove any rows with missing values.
video_games = video_games.dropna(axis=0)

#find correlation
#print video_games.corr()["Global_Sales"]

#Get all the columns from the dataframe.
columns = video_games.columns.tolist()

# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c in ["Critic_Score","NA_Sales","EU_Sales"]]

# Store the variable we'll be predicting on.
target = "Global_Sales"

# Generate the training set.  Set random_state to be able to replicate results.
train = video_games.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the testing set.
test = video_games.loc[~video_games.index.isin(train.index)]

# Initialize the model class.
model = LinearRegression()

# Fit the model to the training data.
model.fit(train[columns], train[target])

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
print mean_squared_error(predictions, test[target])

#save model
filename = 'LinearRegressorModel(video_games)'
pickle.dump(model, open(filename,'wb'))
