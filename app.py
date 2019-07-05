from flask import Flask, request, jsonify
from flask_cors import CORS
from main import Main

import json
import numpy as np

app = Flask(__name__)
CORS(app)

main = Main()
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/file', methods=['POST'])
def file():
  with open("banco_de_audio/teste/test.wav", "wb") as f:
    audio_stream = request.files['file'].read()
    f.write(audio_stream)
  
  voice, names, p = main.evaluateFile('test.wav')
  print(voice)
  p = p.tolist()
  return jsonify({'speaker': voice, 'names': json.dumps(names)
                  , 'p': json.dumps(p)})

@app.route('/model', methods=['POST'])
def model():
  nome = request.form['nomePessoa']
  
  for i in range(0, 1):
    with open("banco_de_audio/treino/"+nome+"-"+str(i)+".wav", "wb") as f:
      audio_stream = request.files['file'].read()
      f.write(audio_stream)
  
  main.readUserData(nome, 1)
  print("PEGOU")
  
  return jsonify({'res': "Seu modelo foi criado!"})
  # voice, names, p = main.evaluateFile('test.wav')
  # print(voice)
  # p = p.tolist()
  # return jsonify({'speaker': voice, 'names': json.dumps(names)
  #                 , 'p': json.dumps(p)})
              