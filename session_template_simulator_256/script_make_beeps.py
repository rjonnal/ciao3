import sys,os
sys.path.append(os.path.split(__file__)[0])
import ciao_config as ccfg
import wave
import math
import struct


#sample_rate = 44100.0
sample_rate = 8000.0

def make_sinewave(
        freq=440.0, 
        duration_seconds=.1, 
        volume=1.0):
    
    out = []
    num_samples = duration_seconds * sample_rate

    for x in range(int(num_samples)):
        out.append(volume * math.sin(2 * math.pi * freq * ( x / sample_rate )))
        
    return out

def save_wav(file_name,buf):
    # Open up a wav file
    wav_file=wave.open(file_name,"w")

    # wav params
    nchannels = 1

    sampwidth = 2

    # 44100 is the industry standard sample rate - CD quality.  If you need to
    # save on file size you can adjust it downwards. The stanard for low quality
    # is 8000 or 8kHz.
    nframes = len(buf)
    comptype = "NONE"
    compname = "not compressed"
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))

    # WAV files here are using short, 16 bit, signed integers for the 
    # sample size.  So we multiply the floating point data we have by 32767, the
    # maximum value for a short integer.  NOTE: It is theortically possible to
    # use the floating point -1.0 to 1.0 data directly in a WAV file but not
    # obvious how to do that using the wave module in python.
    for sample in buf:
        wav_file.writeframes(struct.pack('h', int( sample * 32767.0 )))

    wav_file.close()

    return


try:
    os.mkdir(ccfg.audio_directory)
except Exception as e:
    print(e)

# use middle C scale for basis, and then multipliers (0.125, 0.25, 0.5, 1.0, 2.0, 4.0) for different octaves

octave_multipliers = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
octave_subscripts = [1,2,3,4,5,6]

octaves = dict(list(zip(octave_subscripts,octave_multipliers)))
    
notes = ['C','C_sharp','D','D_sharp','E','F','F_sharp','G','G_sharp','A','A_sharp','B']
freqs = [261.63,277.18,293.66,311.13,329.63,349.23,369.99,392,415.3,440,466.16,493.88]

for subscript in octave_subscripts:
    multiplier = octaves[subscript]
    
    for n,f in zip(notes,freqs):
        fn = os.path.join(ccfg.audio_directory,'%s_%d.wav'%(n,subscript))
        tone = make_sinewave(freq=f*multiplier)
        save_wav(fn,tone)
        print('Writing tone w/ frequency %0.1f to %s'%(f,fn))
