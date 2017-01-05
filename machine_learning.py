#Import required library
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor

#Read in the data.
movies = pandas.read_csv("movie_metadata.csv")

# Remove any rows without gross
movies = movies[movies["gross"] > 0]

# Remove any rows with missing values.
movies = movies.dropna(axis=0)

#find correlation
#print movies.corr()["gross"]

#Get all the columns from the dataframe.
columns = movies.columns.tolist()

# Filter the columns to remove ones we don't want.
columns = [c for c in columns if c in ["num_voted_users"]]#,"num_user_for_reviews", "num_critic_for_reviews"]]

# Store the variable we'll be predicting on.
target = "gross"

# Generate the training set.  Set random_state to be able to replicate results.
train = movies.sample(frac=0.8, random_state=1)

# Select anything not in the training set and put it in the testing set.
test = movies.loc[~movies.index.isin(train.index)]

# Initialize the model class.
model = SGDRegressor(shuffle=True, n_iter=200)

# Fit the model to the training data.
model.fit(train[columns], train[target])

# Generate our predictions for the test set.
predictions = model.predict(test[columns])

# Compute error between our test predictions and the actual values.
print mean_squared_error(predictions, test[target])
