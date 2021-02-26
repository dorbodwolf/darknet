'''
Author: Qinlei
LastEditors: Qinlei
Description:  
Date: 2021-02-25 08:38:57
LastEditTime: 2021-02-25 17:23:38
FilePath: /yolov5-master/main.py
'''

from flask import Flask, render_template,request,jsonify
import sys
import os
from util import base64_to_pil
from gevent.pywsgi import WSGIServer
app = Flask(__name__)
from PIL import Image
sys.path.append("/home/asd/Project/yolov5-master")
@app.route("/")
def Hello():
    message = "hello2"
    return render_template("index.html", temp=message)

# @app.route("/predict", methods=['GET', 'POST'])
# def predict():
#     message = "启动预测程序"
#     print(message)
#     root_path = "/home/asd/Project/yolov5-master/"
#     # os.system(f"python {root_path}test.py --weights {root_path}runs/train/exp9/weights/last.pt  --data {root_path}data/coco_at.yaml --task val --batch-size=12")
    
#     return message
#     # return render_template("index.html", temp=message)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        img_path = "image.png"
        img.save(img_path)

        # Make prediction
        root_path = "/home/asd/Project/yolov5-master/"
        weight_path = "/home/asd/Project/yolov5-master/runs/train/exp9-3bandv1/weights/best.pt"
        code = f"python {root_path}detect.py --source {img_path} --weights {weight_path} --conf 0.25"
        os.system(code)

        result = "sksksksks"
        pred_proba = "0.8"
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=pred_proba)


# app.run(host='0.0.0.0', port=8987,debug=1)

# Serve the app with gevent
http_server = WSGIServer(('0.0.0.0', 8987), app)
http_server.serve_forever()