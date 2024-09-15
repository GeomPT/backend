from flask import Flask, render_template, Response, request
from flask_cors import CORS
from angles import generateFrames, setCurrentMode
from firebase_util import loadFirebaseFromApp, addGraphData, saveFile

app = Flask(__name__)
CORS(app)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed/<mode>")
def video_feed(mode):
    if mode in ["knee", "elbow", "shoulder"]:
        setCurrentMode(mode)
    else:
        setCurrentMode("knee")  # Default mode
    return Response(
        generateFrames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


# Lets the firebase file add api routes
loadFirebaseFromApp(app)

if __name__ == "__main__":
    app.run(debug=True)
