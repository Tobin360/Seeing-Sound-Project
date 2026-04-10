import numpy as np
import matplotlib.image as mpimg
import wave
from array import array


def make_wav(image_filename):
    """ Make a WAV file having a spectrogram resembling an image """
    # Load image
    image = mpimg.imread(image_filename)
    # convert to grayscale
    image = np.sum(image, axis=2)
    # transpose to put pixels into the processing orientation (time x frequency)
    image = image.T
    # flip the frequency axis (y-axis of the spectrogram) so the generated
    # spectrogram keeps the same vertical orientation as the input image
    image = np.flip(image, axis=1)
    image = image**3 
    w, h = image.shape

    # Fourier transform, normalize, remove DC bias
    data = np.fft.irfft(image, h*2, axis=1).reshape((w*h*2))
    data -= np.average(data)
    data *= (2**15-1.)/np.amax(data)
    data = array("h", np.int16(data)).tobytes()

    # Write to disk
    # open file in binary write mode
    output_file = wave.open(image_filename+".wav", "wb")
    output_file.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
    output_file.writeframes(data)
    output_file.close()
    print ("Wrote %s.wav" % image_filename)


if __name__ == "__main__":
    import sys
    make_wav(sys.argv[1])

