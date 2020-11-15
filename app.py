from flask import Flask, render_template, request, jsonify
from NNet import *
import numpy as np
import base64
import re
import base64
import cv2

np.set_printoptions(linewidth=np.inf)
nn = NNet(784, 60, 10)
w1 = np.loadtxt("wt1.txt", delimiter=",")
w2 = np.loadtxt("wt2.txt", delimiter=",")
nn.loadWt1(w1)
nn.loadWt2(w2)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/digitPredict", methods=["POST"])
def digitPredict():
    image_b64 = request.json['imageBase64']
    image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', image_b64))
    image = np.asarray(bytearray(image_data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
    image = ~image

    #Translate to center and resize to fit in 200x200 box
    #...
    
    resized = cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA)
    #print(resized)
    vecIn = np.asarray(resized, dtype="uint8")
    vecIn = vecIn.reshape(28*28).astype('float32')
    predictions = nn.predict(vecIn).tolist()[0]
    
    tot = 0
    largest = 0
    ans = 0
    for i in range(len(predictions)):
        tot += predictions[i]
        if predictions[i] > largest:
            largest = predictions[i]
            ans = i
    probs = [x / tot for x in predictions]
    return jsonify({"winner": ans, "pred": probs})


if __name__ == "__main__":
    app.run(debug = True)
