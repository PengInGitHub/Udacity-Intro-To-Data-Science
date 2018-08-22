import pandas

def filter_by_regular(filename):
    
    turnstile_data = pandas.read_csv(filename)
    turnstile_data = pandas.DataFrame(turnstile_data)
    turnstile_data = turnstile_data[turnstile_data['DESCn'] == 'REGULAR']
    # more of your code here
    return turnstile_data
