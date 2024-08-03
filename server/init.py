from flask import Flask, request, jsonify
from inference import load_model
from djinji import DjinModel

app = Flask(__name__)

djin_model = DjinModel

@app.route('/load-model', methods=['POST'])
def setup_model():
    data = request.get_json()
    if not data : return jsonify(error="No data provided"), 400

    try:
        settings    = data["settings"]

        tempetarure = settings["temperature"]
        max_lenght  = settings["max_lenght"]
        beams       = settings["beams"]

        global djin_model
        djin_model = load_model("../model", max_lenght, beams, tempetarure)
        return f"OK : ok"

    except Exception as e:
        return jsonify(error=f"Err with settings provided. Empty field {e}"), 400

@app.route('/new-question', methods=['POST'])
def new_question():
    question = request.get_json()["question"]
    answer = djin_model.generate_answer(question)
    return f"Received data: {answer}"


if __name__ == '__main__':
    app.run(debug=True)
