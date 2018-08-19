import pandas as pd
import pandasql 

def select_first_50(filename):
 aadhaar = pd.read_csv(filename)
# aadhaar.rename(columns = lambda x: x.replace(' ', '_').lower(), inplace=True)
 aadhaar.rename(columns = lambda x: x.replace(' ','_').lower(), inplace=True)

 q = "select registrar, enrolment_agency from aadhaar limit 5"
 q_complex = "select gender, district, sum(aadhaar_generated) from aadhaar where age>50 group by gender, district"
 aadhaar_sql = pandasql.sqldf(q.lower())
 return aadhaar_sql
    


filename = 'https://s3.amazonaws.com/content.udacity-data.com/courses/ud359/aadhaar_data.csv'
aadhaar = select_first_50(filename)
print aadhaar
