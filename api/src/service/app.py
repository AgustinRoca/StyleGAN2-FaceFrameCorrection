from json_serializer import NumpyArrayEncoder
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

from src.service.service import GeneratorService

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
CORS(app)

service = GeneratorService()

def make_error(status_code, message):
    response = jsonify({
        'status': status_code,
        'message': message
    })
    response.status_code = status_code
    return response

@app.route('/hello')
def home():
    return jsonify({'msg': 'hello! :)'})


@app.route('/faces')
def getFaces():
    id1 = int(request.args.get('id1'))
    id2 = int(request.args.get('id2'))
    imgs_bytes, ids = service.get_images_from_database(id1, id2)
    return jsonify({'ids': ids, 'imgs_bytes': imgs_bytes})

@app.route('/faces', methods=['POST'])
def generateFaces():
    amount = request.form['amount']
    if amount is not None:
        amount = int(amount)
    else:
        amount = 1

    imgs_bytes, zs = service.generate_random_images(amount)
    jsonData = {'zs': zs, 'imgs_bytes': imgs_bytes}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)     

@app.route('/transition', methods=['POST'])
def generateTransition():
    id1 = int(request.form['id1'])
    id2 = int(request.form['id2'])
    amount = int(request.form['amount'])

    imgs_bytes, zs = service.generate_transition(id1, id2, amount)
    jsonData = {'zs': zs, 'imgs_bytes': imgs_bytes}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)    

@app.route('/latentspace', methods=['POST'])
def image_to_latent_space():
    base64str = request.form['file']
    imgs_bytes, zs = service.base64_to_latent(base64str)
    if imgs_bytes is None:
        return make_error(400, 'No face found')
    jsonData = {'zs': zs, 'imgs_bytes': imgs_bytes}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)    

@app.route('/save', methods=['POST'])
def save_latent_code():
    z = request.form['z']
    z = z.split(',')
    z = [float(num) for num in z]
    z = np.array(z)
    z = z.reshape((1,18,512))
    id = service.save_image(z)
    jsonData = {'id': id}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)    

@app.route('/features', methods=['POST'])
def change_features():
    ageAmount = request.form['ageAmount']
    if ageAmount == '':
        ageAmount = 0
    eyeDistanceAmount = request.form['eyeDistanceAmount']
    if eyeDistanceAmount == '':
        eyeDistanceAmount = 0
    eyeEyebrowDistanceAmount = request.form['eyeEyebrowDistanceAmount']
    if eyeEyebrowDistanceAmount == '':
        eyeEyebrowDistanceAmount = 0
    eyeRatioAmount = request.form['eyeRatioAmount']
    if eyeRatioAmount == '':
        eyeRatioAmount = 0
    eyesOpenAmount = request.form['eyesOpenAmount']
    if eyesOpenAmount == '':
        eyesOpenAmount = 0
    genderAmount = request.form['genderAmount']
    if genderAmount == '':
        genderAmount = 0
    lipRatioAmount = request.form['lipRatioAmount']
    if lipRatioAmount == '':
        lipRatioAmount = 0
    mouthOpenAmount = request.form['mouthOpenAmount']
    if mouthOpenAmount == '':
        mouthOpenAmount = 0
    mouthRatioAmount = request.form['mouthRatioAmount']
    if mouthRatioAmount == '':
        mouthRatioAmount = 0
    noseMouthDistanceAmount = request.form['noseMouthDistanceAmount']
    if noseMouthDistanceAmount == '':
        noseMouthDistanceAmount = 0
    noseRatioAmount = request.form['noseRatioAmount']
    if noseRatioAmount == '':
        noseRatioAmount = 0
    noseTipAmount = request.form['noseTipAmount']
    if noseTipAmount == '':
        noseTipAmount = 0
    pitchAmount = request.form['pitchAmount']
    if pitchAmount == '':
        pitchAmount = 0
    rollAmount = request.form['rollAmount']
    if rollAmount == '':
        rollAmount = 0
    smileAmount = request.form['smileAmount']
    if smileAmount == '':
        smileAmount = 0
    yawAmount = request.form['yawAmount']
    if yawAmount == '':
        yawAmount = 0
    id = request.form['id']
    features_dict = {}
    
    features_dict['age'] = float(ageAmount)
    features_dict['eye_distance'] = float(eyeDistanceAmount)
    features_dict['eye_eyebrow_distance'] = float(eyeEyebrowDistanceAmount)
    features_dict['eye_ratio'] = float(eyeRatioAmount)
    features_dict['eyes_open'] = float(eyesOpenAmount)
    features_dict['gender'] = float(genderAmount)
    features_dict['lip_ratio'] = float(lipRatioAmount)
    features_dict['mouth_open'] = float(mouthOpenAmount)
    features_dict['mouth_ratio'] = float(mouthRatioAmount)
    features_dict['nose_mouth_distance'] = float(noseMouthDistanceAmount)
    features_dict['nose_ratio'] = float(noseRatioAmount)
    features_dict['nose_tip'] = float(noseTipAmount)
    features_dict['pitch'] = float(pitchAmount)
    features_dict['roll'] = float(rollAmount)
    features_dict['smile'] = float(smileAmount)
    features_dict['yaw'] = float(yawAmount)

    img_bytes, z = service.change_features(id, features_dict)
    jsonData = {'z': z, 'img_bytes': img_bytes}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)  

@app.route('/interchange', methods=['POST'])
def mix_styles():
    id1 = int(request.form['id1'])
    id2 = int(request.form['id2'])

    imgs_bytes, zs = service.mix_styles(id1, id2)
    jsonData = {'zs': zs, 'imgs_bytes': imgs_bytes}
    return json.dumps(jsonData, cls=NumpyArrayEncoder)     


if __name__ == '__main__':
    app.run('0.0.0.0', 5000)
