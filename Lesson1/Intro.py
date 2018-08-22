import pandas as pd
import numpy as np

if False:
    train =pd.read_csv("Berkeley.csv")
    print train['Admit'][train['Freq']>=100]

#pandas series
if False:
    series = pd.Series(['Dave', 'Bo-Han', 'Udacity', 42, 9600],
                        index = ['Instructor', 'Course Manager', 'Platform', 'Cost','Subscribe'])
    print series[['Instructor','Platform']]

if False:
    cuteness = pd.Series([1, 2, 3, 4, 5],
                        index = ['Cockroach', 'Fish', 'Mini Pig', 'Puppy', 'Kitten'])
    print cuteness[cuteness>3]
    df = pd.DataFrame(cuteness)
    print df.loc['Cockroach']#use df.loc['indexName'] to retrieve row

#data frame, use dict key the name of column, value the list of values
if False:
    data = {"year":[2010, 2011, 2012, 2011, 2012, 2010, 2011, 2012],
            "team":['Bears', 'Bears', 'Bears', 'Packers', 'Packers', 'Lions', 'Lions', 'Lions'],
            "wins":[11, 8, 10, 15, 11, 6, 10, 4],
            "losses":[5, 8, 6, 1, 5, 10, 6, 12]}
    football = pd.DataFrame(data)
    print football.dtypes
    print football.describe()
    print football.head()
    print football.tail()
    print football.year
    print football['year']
    print football[['year', 'wins', 'team']]
    print football.iloc[[0]]
    print football.loc[[0]]
    print football[3:5]
    print football[football.wins>3]
    print football[football['wins']>3]
    print football[(football.wins>10)&(football.team=='Bears')]


from pandas import DataFrame, Series

#################
# Syntax Reminder:
#
# The following code would create a two-column pandas DataFrame
# named df with columns labeled 'name' and 'age':
#
# people = ['Sarah', 'Mike', 'Chrisna']
# ages  =  [28, 32, 25]
# df = DataFrame({'name' : Series(people),
#                 'age'  : Series(ages)})

def create_dataframe():
    '''
    Create a pandas dataframe called 'olympic_medal_counts_df' containing
    the data from the table of 2014 Sochi winter olympics medal counts.  

    The columns for this dataframe should be called 
    'country_name', 'gold', 'silver', and 'bronze'.  

    There is no need to  specify row indexes for this dataframe 
    (in this case, the rows will automatically be assigned numbered indexes).
    
    You do not need to call the function in your code when running it in the
    browser - the grader will do that automatically when you submit or test it.
    '''

    countries = ['Russian Fed.', 'Norway', 'Canada', 'United States',
                 'Netherlands', 'Germany', 'Switzerland', 'Belarus',
                 'Austria', 'France', 'Poland', 'China', 'Korea', 
                 'Sweden', 'Czech Republic', 'Slovenia', 'Japan',
                 'Finland', 'Great Britain', 'Ukraine', 'Slovakia',
                 'Italy', 'Latvia', 'Australia', 'Croatia', 'Kazakhstan']

    gold = [13, 11, 10, 9, 8, 8, 6, 5, 4, 4, 4, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    silver = [11, 5, 10, 7, 7, 6, 3, 0, 8, 4, 1, 4, 3, 7, 4, 2, 4, 3, 1, 0, 0, 2, 2, 2, 1, 0]
    bronze = [9, 10, 5, 12, 9, 5, 2, 1, 5, 7, 1, 2, 2, 6, 2, 4, 3, 1, 2, 1, 0, 6, 2, 1, 0, 1]

    # your code here
    medal_counts = {'country_name':pd.Series(countries),
                    'gold':pd.Series(gold),
                    'silver':pd.Series(silver),
                    'bronze':pd.Series(bronze)}
    olympic_medal_counts_df = pd.DataFrame(medal_counts)
    return olympic_medal_counts_df

data = create_dataframe()
if True:
    print data[data['gold']>=9]#slice data frame by cell value
    print data['gold'].mean()
    print data['gold'].map(lambda x:x>=9)
    data['medal_avg'] =  data.apply(lambda row: (row.gold+row.silver+row.bronze)/3, axis=1)
    print data
    series = np.dot(data[['gold','silver','bronze']],[4,2,1])                  
    olympic_points = {'country_name': data['country_name'],
                      'points': pd.Series(series)}
    olympic_points_df = DataFrame(olympic_points)
    print olympic_points_df
