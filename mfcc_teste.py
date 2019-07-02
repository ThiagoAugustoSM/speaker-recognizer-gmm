from python_speech_features import mfcc
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np

entrada = input("Digite um nome:" )
nome_audio = entrada + '.wav'
(rate,sig) = wav.read(nome_audio)
mfcc_feat = mfcc(sig,rate)

teste = np.array(mfcc_feat)
arq = open(entrada + '2.txt', 'a')
arq.write('[')

for i in range(0,len(teste)-1):
	arq.write(str(teste[i]))
	arq.write(str(','))
arq.write(str(teste[i]))
arq.write(']')
arq.close()