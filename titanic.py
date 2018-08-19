import numpy
import pandas as pd
import statsmodels.api as sm

def simple_heuristic(file_path):
    
    predictions = {}
    df = pd.read_csv(file_path)
    for _, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        if passenger['Sex'] == 'male':
            predictions[passenger_id] = 0
        elif passenger['Sex'] == 'female':
            predictions[passenger_id] = 1
            
        
    return predictions


def complex_heuristic(file_path):

    predictions = {}
    df = pd.read_csv(file_path)
    for _, passenger in df.iterrows():
        passenger_id = passenger['PassengerId']
        if passenger['Sex'] == 'female' or (passenger['Pclass'] == 1 and passenger['Age'] < 18):
            predictions[passenger_id] = 1
        else:
            predictions[passenger_id] = 0

    return predictions

file_path = 'titanic_data.csv'
titanic = pd.read_csv(file_path)
titanic = pd.DataFrame(titanic)

#print titanic[['Name','Pclass']][(titanic['Sex']=='male')&(~titanic['Name'].str.contains('Mr.'))]
print titanic['Cabin']
# predictions = complex_heuristic(file_path)
# print predictions
