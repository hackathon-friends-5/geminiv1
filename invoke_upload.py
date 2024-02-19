import requests

# URL of the Flask API
url = 'http://127.0.0.1:5000/upload'

# List of PDF files to upload
files = [
    ('files', open('DS - 8.pdf', 'rb'))
]

# Send the POST request
response = requests.post(url, files=files)

# Print the response
print(response.json())


