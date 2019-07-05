# Math modules
import numpy as np

# Utils
import os
import scipy.io.wavfile as wav
import pickle #Serialize objects

# Feature Extraction
from python_speech_features import mfcc

# Classifier
import sklearn.mixture

class Main:

  def __init__(self):
    self.source = 'banco_de_audio/'
    self.dest = 'gmmModels/'
    self.gmmModelsFiles = os.listdir(self.dest)
    self.gmmModels = [pickle.load(open(self.dest + file, 'rb')) for file in self.gmmModelsFiles]
    print(self.gmmModelsFiles)

  def extractFeatures(self, path):
    (rate, sig) = wav.read(path)
    return mfcc(sig, rate)

  def readUserData(self, userName, qntFiles):

    filePaths = []
    for i in range(0, qntFiles):
      filePaths.append(self.source + 'treino/' +
                        userName + '-' + str(i) + '.wav')

    features = np.asarray(())
    for path in filePaths:
      mfcc_feat = self.extractFeatures(path)
      if(features.size == 0):
        features = np.array(mfcc_feat)
      else:
        features = np.vstack((features, np.array(mfcc_feat)))
    self.createGMMModel(features, userName)

  def createGMMModel(self, features, userName):
    gmm = sklearn.mixture.GaussianMixture(n_components = 16, covariance_type = 'diag', n_init = 3)
    gmm.fit(features)

    pickle.dump(gmm, open(self.dest + userName + '.gmm', 'wb'))
    
    self.gmmModelsFiles = os.listdir(self.dest)
    self.gmmModels = [pickle.load(open(self.dest + file, 'rb')) for file in self.gmmModelsFiles]
    
    print(userName + ' GMM Model created succesfully!')

  def evaluateTest(self):
    
    filePaths = os.listdir(self.source + 'teste')
    for filePath in filePaths:
      vector = self.extractFeatures(self.source + 'teste/' + filePath)
      log_likelihood = np.zeros(len(self.gmmModels))
      for i in range(len(self.gmmModels)):
        gmm    = self.gmmModels[i]  #checking with each model one by one
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

      print(log_likelihood)
      winner = np.argmax(log_likelihood)
      norma = 1/log_likelihood
      norma = -norma/np.linalg.norm(-norma)
      print(norma)
      # print(self.log_softmax(norma))

      print("WINNER", winner)
      print(filePath, ' winner was: ', self.gmmModelsFiles[winner].split('.')[0]) 
  
  def evaluateFile(self, fileName):
    vector = self.extractFeatures(self.source + 'teste/' + fileName)
    log_likelihood = np.zeros(len(self.gmmModels))
    for i in range(len(self.gmmModels)):
      gmm    = self.gmmModels[i]  #checking with each model one by one
      scores = np.array(gmm.score(vector))
      log_likelihood[i] = scores.sum()

    winner = np.argmax(log_likelihood)
    p = 1/log_likelihood
    p = -p/np.linalg.norm(-p)
    print(fileName, ' winner was: ', self.gmmModelsFiles[winner].split('.')[0]) 
    return self.gmmModelsFiles[winner].split('.')[0], self.gmmModelsFiles, p

  def log_softmax(self, x):
    e_x = np.exp(x - np.max(x))
    return np.log(e_x / e_x.sum())

main = Main()

if __name__ == "__main__":
  print("BEM VINDO AO RECONHECEDOR DE VOZ!")
  print("O QUE DESEJA FAZER? ")
  print("1 - INDENTIFICAR UM USUÁRIO DO BANCO")
  print("2 - RODAR A ROTINA DE TESTE COM OS MODELOS PREVIOS DE GMM")
  choice = input(': ')
  if choice == '1':
    id = input("ID DO USUARIO: ")
    main.readUserData(id, 3)
  elif choice == '2':
    main.evaluateTest()