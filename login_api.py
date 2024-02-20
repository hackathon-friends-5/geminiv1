from flask import Flask, request, jsonify


app = Flask(__name__)


@app.route('/login', methods=['POST'])
def login():
    username = request.json['username']
    password = request.json['password']
    additional_data = request.json['additional_data']
    
    # if not username or not password:
    #     return jsonify({'message': 'Username and password are required.'}), 400


    # save(username, password, additional_data)

    return jsonify({'message': additional_data})


def save(username, password, additional_data):

    user = {'username':username,'password':password,'data':additional_data}
    with open(r'users.txt', 'a') as file_write:
        file_write.write(user)


if __name__ == '__main__':
    app.run(port=5000)
