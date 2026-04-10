"""visualize.py

Read a mono WAV file and produce a spectrogram image using matplotlib.

Usage:
    python visualize.py input.wav         # shows the spectrogram window (if available)
    python visualize.py input.wav -o out.png  # saves spectrogram to PNG without showing

This script uses only standard library + numpy/matplotlib which are already required by the project.
"""
import argparse
from pathlib import Path
import wave
from array import array
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_wav(path):
    """Load mono WAV and return (sample_rate, numpy array float in [-1,1])"""
    with wave.open(str(path), 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError('visualize.py expects 16-bit WAV files (sampwidth=2)')

    arr = array('h')  # signed 16-bit
    arr.frombytes(raw)
    samples = np.frombuffer(arr, dtype=np.int16).astype(np.float32)

    if n_channels > 1:
        # convert to mono by averaging channels
        samples = samples.reshape((-1, n_channels)).mean(axis=1)

    samples /= 2**15  # -1..1
    return framerate, samples


def plot_spectrogram(rate, data, outfile=None, cmap='viridis', nfft=2048, noverlap=None):
    plt.figure(figsize=(10, 5))
    if noverlap is None:
        noverlap = nfft // 2

    # matplotlib.pyplot.specgram returns (spectrum, freqs, bins, im)
    Pxx, freqs, bins, im = plt.specgram(
        data, NFFT=nfft, Fs=rate, noverlap=noverlap, cmap=cmap, scale='dB'
    )
    im.set_clim(-120, -20)  # optional: sets a consistent visible dB range

    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Power spectral density (dB)')
    plt.tight_layout()

    if outfile:
        plt.savefig(str(outfile), dpi=150)
        print('Wrote', outfile)
        plt.close()
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description='Visualize a WAV file as a spectrogram (no SoX required).')
    p.add_argument('wav', help='input WAV file')
    p.add_argument('-o', '--output', help='save output PNG file instead of showing')
    p.add_argument('--colormap', default='viridis', help='matplotlib colormap name')
    p.add_argument('--nfft', type=int, default=2048, help='FFT window size')
    p.add_argument('--noverlap', type=int, default=None, help='overlap between windows (samples)')

    args = p.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        raise SystemExit('Input WAV does not exist: %s' % wav)

    rate, samples = load_wav(wav)
    plot_spectrogram(rate, samples, outfile=args.output, cmap=args.colormap, nfft=args.nfft, noverlap=args.noverlap)


if __name__ == '__main__':
    main()
