import os
import numpy as np
import scipy
from scipy.io import wavfile
import scipy.fftpack as fft
from scipy.signal import get_window
#import IPython.display as ipd
import matplotlib.pyplot as plt

#Informações:
#O programa recebe uma entrada que o nome do arquivo de audio da pasta banco_de_audio
#Executa operações relacionadas a MFCC
#Salva em um arquivo TXT de mesmo nome que o audio selecionado anteriormente o resultado da MFCC, um array [[],...[],...[]...] para
#ser o input da GMM 


def normalize_audio(audio):
    audio = audio / np.max(np.abs(audio))
    return audio

#Verificar SAMPLES, pois ó utilizado aqui é de 44100, mas o utilizado na aplicação final poderá ter um valor diferente
def frame_audio(audio, FFT_size=2048, hop_size=10, sample_rate=16000):
    # hop_size in ms
    
    audio = np.pad(audio, int(FFT_size / 2), mode='reflect')
    frame_len = np.round(sample_rate * hop_size / 1000).astype(int)
    frame_num = int((len(audio) - FFT_size) / frame_len) + 1
    frames = np.zeros((frame_num,FFT_size))
    
    for n in range(frame_num):
        frames[n] = audio[n*frame_len:n*frame_len+FFT_size]
    
    return frames

def freq_to_mel(freq):
    
    return 2595.0 * np.log10(1.0 + freq / 700.0)

def met_to_freq(mels):
    
    return 700.0 * (10.0**(mels / 2595.0) - 1.0)

def get_filter_points(fmin, fmax, mel_filter_num, FFT_size, sample_rate=16000): # sample_rate estava com 44100
    fmin_mel = freq_to_mel(fmin)
    fmax_mel = freq_to_mel(fmax)
    
    print("MEL min: {0}".format(fmin_mel))
    print("MEL max: {0}".format(fmax_mel))
    
    mels = np.linspace(fmin_mel, fmax_mel, num=mel_filter_num+2)
    freqs = met_to_freq(mels)
    
    return np.floor((FFT_size + 1) / sample_rate * freqs).astype(int), freqs

def get_filters(filter_points, FFT_size):
    filters = np.zeros((len(filter_points)-2,int(FFT_size/2+1)))
    
    for n in range(len(filter_points)-2):
        filters[n, filter_points[n] : filter_points[n + 1]] = np.linspace(0, 1, filter_points[n + 1] - filter_points[n])
        filters[n, filter_points[n + 1] : filter_points[n + 2]] = np.linspace(1, 0, filter_points[n + 2] - filter_points[n + 1])
    
    return filters

def dct(dct_filter_num, filter_len):
    basis = np.empty((dct_filter_num,filter_len))
    basis[0, :] = 1.0 / np.sqrt(filter_len)
    
    samples = np.arange(1, 2 * filter_len, 2) * np.pi / (2.0 * filter_len)

    for i in range(1, dct_filter_num):
        basis[i, :] = np.cos(i * samples) * np.sqrt(2.0 / filter_len)
        
    return basis


#---------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":

	entrada = input("Nome do audio: ")
	nome_audio = "banco_de_audio/" + entrada + ".wav"
	samples , audio = wavfile.read(nome_audio)
	print("Samples: {0}Hz".format(samples))
	print("Audio duration: {0}s".format(len(audio) / samples))

	plt.figure(figsize=(15,4))
	plt.plot(np.linspace(0, len(audio) / samples, num=len(audio)), audio)
	plt.grid(True)
	plt.show()

	#Trazendo a faixa de audio para a primeira região de Nyquist (0 - 22,05KHz)
	audio = normalize_audio(audio)
	plt.figure(figsize=(15,4))
	plt.plot(np.linspace(0, len(audio) / samples, num=len(audio)), audio)
	plt.grid(True)
	plt.show()

	#Como o audio é um processo não estacionario, ao aplicar a FFT isso irá causar problemas(distorções), por isso, utilizamos de uma tecnica 
	#	que consiste em dividir o audio em diversos frames (trechos), para que a analise possa ocorrer de forma "estacionaria",
	hop_size = 15 #ms
	FFT_size = 2048

	frames = frame_audio(audio, FFT_size=FFT_size, hop_size=hop_size, sample_rate=samples)
	print("Indica em quantos frames foi separado o audio e quantas 'features' há em cada frame: {0}".format(frames.shape))


	#Como isso pôde-se fazer com que se perca algumas informações de borda, é importante que esses trechos se sobreponham para isso
	#	eles precisam ter uma correlação, que é realizada com uma função window, que garante que as extremidades possam terminar com valores
	#	proximos a zero, suavizando as extremiadas de cada frame, e a função utilizada é a Hanning, identificada no parametro 'hann'.

	window = get_window("hann", FFT_size, fftbins=True)
	plt.figure(figsize=(15,4))
	plt.plot(window)
	plt.grid(True)
	plt.show()

	#E então podemos converter o audio para o dominio da frequencia
	audio_frequencia = frames * window
	'''
	for i in range(0,1):
		plt.figure(figsize=(15,6))
		plt.subplot(2, 1, 1)
		plt.plot(frames[i])
		plt.title('Original Frame')
		plt.grid(True)
		plt.subplot(2, 1, 2)
		plt.plot(audio_frequencia[i])
		plt.title('Frame After Windowing')
		plt.grid(True)
		plt.show()
	'''

	#Podemos aplicar agora a FFT e pega-se somente a parte positiva ( half -> +1 ) 
	audio_winT = np.transpose(audio_frequencia)
	audio_fft = np.empty((int(1 + FFT_size // 2), audio_winT.shape[1]), dtype=np.complex64, order='F')

	for n in range(audio_fft.shape[1]):
	    audio_fft[:, n] = fft.fft(audio_winT[:, n], axis=0)[:audio_fft.shape[0]]

	audio_fft = np.transpose(audio_fft)
	audio_power = np.square(np.abs(audio_fft))
	print(audio_power.shape)

	#Criamos então um banco de filtros de espaçamento MEL, para que o audio possa ser "passado" atraves dele, e podemos então obter a potencia
	#	em cada um dos frames de frequencia do audio
	#O espaçamento entre os filtros cresce exponencialmente com a frequencia
	freq_min = 0
	freq_high = samples / 2
	mel_filter_num = 10

	print("Minimum frequency: {0}".format(freq_min))
	print("Maximum frequency: {0}".format(freq_high))


	#Precisamos definir os pontos para os limites do filtro com o espaçamento na escala MEL, e então contruimos um array linearmente espaçado
	#	entre as duas frequencias MEL, depois convertemos a matriz para o espaço da frequencia e normalizamos o tamanho dela da FFT e escolhemos
	#	seus valores.

	filter_points, mel_freqs = get_filter_points(freq_min, freq_high, mel_filter_num, FFT_size, sample_rate=16000)
	print (filter_points)

	#Como os pontos de filtro já foram definidos, contruimos agora os filtros 
	filters = get_filters(filter_points, FFT_size)

	'''
	plt.figure(figsize=(15,4))
	for n in range(filters.shape[0]):
		plt.title('Filter 1')
		plt.plot(filters[n])
		plt.show()
	'''

	#Divimos então os triangulos de MEL pela largura da banda MEL (fazendo então uma normalização da área), esta ação é necessaria para que se
	#	evite que o ruído aumente com a frequencia

	enorm = 2.0 / (mel_freqs[2:mel_filter_num+2] - mel_freqs[:mel_filter_num])
	filters *= enorm[:, np.newaxis]

	'''
	plt.figure(figsize=(15,4))
	for n in range(filters.shape[0]):
		plt.title('Filter 2')
		plt.plot(filters[n])
		plt.show()
	'''


	#Agora obtemos a matriz representando a potencia de audio em todos os 10 filtros em diferentes intervalos de tempo
	audio_filtered = np.dot(filters, np.transpose(audio_power))
	audio_log = 10.0 * np.log10(audio_filtered)
	print (audio_log.shape)


	#O ultimo passo para se obter o resultado da MFCC é aplicar a DCT 3 (Transforada Discreta do Cosseno), possibilitando que seja
	#	extraido alterações de alta e baixa frequencia do sinal.

	dct_filter_num = 40
	dct_filters = dct(dct_filter_num, mel_filter_num)

	cepstral_coefficents = np.dot(dct_filters, audio_log)
	#print ("Valor de Cepstral:",cepstral_coefficents)

	plt.figure(figsize=(15,5))
	plt.plot(np.linspace(0, len(audio) / samples, num=len(audio)), audio)
	plt.imshow(cepstral_coefficents, aspect='auto', origin='lower');


	#Salvando as 'features' em um arquivo txt
	teste = np.array(cepstral_coefficents)
	arq = open("features_txt/" + entrada + '.txt', 'a')
	arq.write('[')

	for i in range(0,len(teste)-1):
		arq.write(str(teste[i]))
		arq.write(str(','))
	arq.write(str(teste[i]))
	arq.write(']')
	arq.close()

#https://www.kaggle.com/ilyamich/mfcc-implementation-and-tutorial