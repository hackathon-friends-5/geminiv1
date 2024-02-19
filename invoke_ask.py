import requests

# URL of the Flask API
url = 'http://127.0.0.1:5000/ask'

# JSON data containing the question
question_data = {'question': 'summarize the given doc.'}

# Send the POST request
response = requests.post(url, json=question_data)

# Print the response
print(response.json())