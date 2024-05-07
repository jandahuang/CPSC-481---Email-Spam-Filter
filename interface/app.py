from flask import Flask
from flask import render_template
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/email", methods=["post"])
def email():
    # This is where we call the spam filter and return the result in json
    return jsonify({
            "response": "not spam",
            "tests": {
                "logistic_regression": "not spam",
                "naive_bayes": "spam",
                "decision_tree": "not spam",
                "support_vector_machine": "not spam",
                "k_nearest_neighbors": "not spam",
                "random_forest": "spam"
            }
        })