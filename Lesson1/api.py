import requests
import json

def api_call():
    url = 'https://github.com/timeline.json'
    data = requests.get(url).text
    data = json.loads(data)
    return data['message']

data = api_call()
print data
