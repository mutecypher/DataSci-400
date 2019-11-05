"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# Make the pandas package available
import pandas as pd

# Create an empty data frame called Cars
Cars = pd.DataFrame()

# View the data frame
Cars.head()
#################

# Create a column of price categories that has values for the first 4 cars
Cars.loc[:,"buying"]  = ["vhigh", "high", "low", "med"]

# Create a column of number of doors that has values for the first 4 cars
Cars.loc[:,"doors"]  = [2, 2, 4, 4]

# View the data frame
Cars.head()
##################

# Add a fifth row of data
Cars.loc[4]  = ["vhigh", 3]

# View the data frame
Cars.head()
##################

# View the data types of the columns in the data frame
Cars.dtypes

####################

## Stuff for the quiz

Vehicle = pd.DataFrame()
Vehicle.loc[:,"Type"] = ["Tricycle", "Car", "Motorcycle"]
Wheels = [3, 4, 2]
Vehicle.loc[:,"Wheels"] = Wheels
Vehicle
Vehicle.loc[3, :] = ["Sled", 0]
Vehicle.loc[:, "Doors"]  = [0, 2, 0, 0]
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
Adult = pd.read_csv(url)
Adult = pd.read_csv(url, header=None)
###

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data"
BloodDonation = pd.read_csv(url)
##


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data"
Votes = pd.read_csv(url, header=None)
Faps = (Votes != "?")
Attempt = Votes[Faps]
0+0+12+48+11+11+15+11+14+15+22+7+21+31+25+17+28
