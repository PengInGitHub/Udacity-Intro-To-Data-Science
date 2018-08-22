import pandas

def get_hourly_entries(df):
    #https://stackoverflow.com/questions/23142967/adding-a-column-thats-result-of-difference-in-consecutive-rows-in-pandas
    #https://stackoverflow.com/questions/38134012/pandas-dataframe-fillna-only-some-columns-in-place
    df['ENTRIESn_hourly'] = df['ENTRIESn'] - df['ENTRIESn'].shift(1)
    df['ENTRIESn_hourly'] = df['ENTRIESn_hourly'].fillna(value=1)
    return df
