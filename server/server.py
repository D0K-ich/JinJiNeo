from flask import Flask, request

app = Flask(__name__)

@app.route('/send-message', methods=['POST'])
def data():
    json_data = request.get_json()
    print(json_data)
    return f"Received data: {json_data}"


if __name__ == '__main__':
    app.run(debug=True)
