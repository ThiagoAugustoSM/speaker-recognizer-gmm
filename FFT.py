#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Site de referencia
#https://stackoverflow.com/questions/23377665/python-scipy-fft-wav-files3

from __future__ import print_function
import scipy.io.wavfile as wavfile
import scipy
import scipy.fftpack
import numpy as np
from matplotlib import pyplot as plt

#Leitura do arquivo wav
entrada = input("Nome do audio: ")
nome_audio = "banco_de_audio/" + entrada + ".wav"
fs_rate, audio = wavfile.read(nome_audio)
print ("Frequency sampling", fs_rate)
#-------------------------------------------
#Caso o áudio tenha 2 canais, ele divide pegando apenas o canal da esquerda.
l_audio = len(audio.shape)
if l_audio == 2:
    audio = audio.sum(axis=1) / 2
#-------------------------------------------

#Frame Blocking
#-------------------------------------------
N = audio.shape[0] # Quantidade de amostras
print ("Complete Samplings N", N)
secs = N / float(fs_rate) # Quantos segundos tem o áudio
Ts = 1.0/fs_rate #Período entre amostragens
print ("Timestep between samples Ts", Ts)
t = scipy.arange(0, secs, Ts) # Array com os instantes de tempo que foram tiradas as amostras
#-------------------------------------------

#Windowing
#-------------------------------------------
hann = np.hanning(len(audio)) # Hanning, ação que faz o window

plt.subplot(311)
plt.title('Original Audio')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.plot(t,audio)
plt.subplot(312)
plt.title('Audio with Hanning Window function applied')
plt.xlabel('Time ($s$)')
plt.ylabel('Amplitude')
plt.plot(t,hann*audio)
plt.show() # Comparação entre o áudio com e sem window


#-------------------------------------------

#FFT
#------------------------------------------
FFT = abs(scipy.fft(audio*hann))
#------------------------------------------- Utilizando apenas a parte positiva da FFT
m = range(N//2)
FFT_side = FFT[m] # one side FFT range
print ("FFT_side: ",FFT_side)

freqs = scipy.fftpack.fftfreq(audio.size, t[1]-t[0])
fft_freqs = np.array(freqs)
freqs_side = freqs[range(N//2)] 
fft_freqs_side = np.array(freqs_side) 
#------------------------------------------

#Plotando graficos
#------------------------------------------
plt.subplot(312)
p2 = plt.plot(freqs, FFT, "r") # plotting the complete fft spectrum

plt.title('Audio in FFT Frequency')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count dbl-sided')
plt.subplot(313)
p3 = plt.plot(freqs_side, abs(FFT_side), "b") # plotting the positive fft spectrum
plt.title('Audio in FFT Single Sided')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Count single-sided')
plt.show()
#------------------------------------------

#Salvando as 'features' em um arquivo txt
arq = open("features_txt/" + entrada + 'FFT.txt', 'a')
arq.write('[')
for i in range(0,len(FFT)-1):
	arq.write(str(round(FFT[i],4)))
	arq.write(str(','))
arq.write(str(round(FFT[i],4)))
arq.write(']')
arq.close()

'''
if __name__ == "__main__":
	sound = "audio.wav"
	f(sound)
'''