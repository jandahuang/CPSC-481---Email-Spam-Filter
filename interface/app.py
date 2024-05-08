from flask import Flask
from flask import render_template
from flask import jsonify
from flask import request
from spam_filter import spam_filter

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/email", methods=["post"])
def email():
    data = request.get_json()
    # This is where we call the spam filter and return the result in json
    return jsonify(spam_filter(data["email"]))

