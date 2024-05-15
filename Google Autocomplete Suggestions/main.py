import json
import requests

completion_query = 'Real Madrid'

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0" # Just google your current User-Agent and replace it here
}

response = requests.get(f'http://google.com/complete/search?client=chrome&q={completion_query}')

for completion in json.loads(response.text)[1]:
    print(completion)