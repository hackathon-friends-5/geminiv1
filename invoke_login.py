import requests

url = 'https://6b5e-115-242-70-50.ngrok-free.app/login'

import json 

data = {
    'username': 'example_username',
    'password': 'example_password',
    'additional_data': 'example_additional_data'
}


response = requests.post(url, json=data)

print(response.json())


