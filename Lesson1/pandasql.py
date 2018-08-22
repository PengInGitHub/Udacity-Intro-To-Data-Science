import pandas as pd
import pandasql 

def select_first_50(filename):
 aadhaar = pd.read_csv(filename)
# aadhaar.rename(columns = lambda x: x.replace(' ', '_').lower(), inplace=True)
 aadhaar.rename(columns = lambda x: x.replace(' ','_').lower(), inplace=True)

 q = "select registrar, enrolment_agency from aadhaar limit 5"
 q_complex = "select gender, district, sum(aadhaar_generated) from aadhaar where age>50 group by gender, district"
 q_weather = "select count(*) from weather_data where cast( rain as integer) = 1"
 q_maxtemp = "select fog, max(max_temp) from weather_data group by fog"
 q_meantemp = "select avg(meantemp) from weather_data where cast(strftime('%w', date) as integer)=0 or cast(trftime('%w', date) as integer)=6"
 q_rain = "select avg(mintempi) from weather_data where rain=1 and mintempi>55"

 aadhaar_sql = pandasql.sqldf(q.lower())
 return aadhaar_sql
    


filename = 'https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/aadhaar_data.csv'
aadhaar = select_first_50(filename)
print aadhaar
