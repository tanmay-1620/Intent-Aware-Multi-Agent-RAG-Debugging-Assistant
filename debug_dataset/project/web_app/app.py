from flask import Flask, request, jsonify

app = Flask(__name__)

users = {"admin": "1234"}

@app.route("/login", methods=["POST"])
def login():
    data = request.json
    if data["username"] in users:
        if data["password"] == users[data["username"]]:
            return jsonify({"status": "success"})
    return jsonify({"status": "failed"})

if __name__ == "__main__":
    app.run(debug=True)