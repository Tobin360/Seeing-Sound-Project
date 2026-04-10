import numpy as np
import matplotlib.image as mpimg
import wave
from array import array


# -------------------------
# WAV I/O (16-bit PCM mono)
# -------------------------
def load_wav_mono(path):
    """Load a WAV file and return (sample_rate, float32 samples in [-1, 1])."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Expected 16-bit WAV (sampwidth=2). Got sampwidth=%d" % sampwidth)

    arr = array("h")
    arr.frombytes(raw)
    x = np.frombuffer(arr, dtype=np.int16).astype(np.float32)

    if n_channels > 1:
        x = x.reshape((-1, n_channels)).mean(axis=1)

    x /= 2**15  # -1..1
    return rate, x


def write_wav_mono(path, rate, x):
    """Write float samples in [-1,1] to a 16-bit mono WAV."""
    x = np.asarray(x, dtype=np.float32)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * (2**15 - 1)).astype(np.int16).tobytes()

    with wave.open(str(path), "wb") as wf:
        wf.setparams((1, 2, int(rate), 0, "NONE", "not compressed"))
        wf.writeframes(pcm)


# -------------------------
# Image -> watermark mask
# -------------------------
def _to_grayscale01(img):
    """Convert loaded image to grayscale float32 in [0,1]."""
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 2:
        g = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        g = img[..., :3].mean(axis=2)
    else:
        raise ValueError("Unsupported image shape: %r" % (img.shape,))
    g -= g.min()
    if g.max() > 1e-8:
        g /= g.max()
    return g


def _dilate_max(img, k=3, iters=1):
    """Simple grayscale dilation via max filter (no scipy)."""
    if iters <= 0:
        return img
    pad = k // 2
    out = img
    for _ in range(iters):
        p = np.pad(out, ((pad, pad), (pad, pad)), mode="edge")
        acc = np.zeros_like(out)
        # sliding max (k is small; loops are OK for these image sizes)
        for dy in range(k):
            for dx in range(k):
                acc = np.maximum(acc, p[dy:dy + out.shape[0], dx:dx + out.shape[1]])
        out = acc
    return out


def _blur_separable(img, passes=1):
    """Light 1D blur in x and y using kernel [0.25, 0.5, 0.25]."""
    if passes <= 0:
        return img
    k = np.array([0.25, 0.5, 0.25], dtype=np.float32)
    out = img.astype(np.float32, copy=True)

    def conv1d_axis(a, axis):
        pad = 1
        p = np.pad(a, [(pad, pad) if i == axis else (0, 0) for i in range(a.ndim)], mode="edge")
        # three-tap convolution
        s0 = np.take(p, indices=range(0, a.shape[axis]), axis=axis)
        s1 = np.take(p, indices=range(1, a.shape[axis] + 1), axis=axis)
        s2 = np.take(p, indices=range(2, a.shape[axis] + 2), axis=axis)
        return k[0] * s0 + k[1] * s1 + k[2] * s2

    for _ in range(passes):
        out = conv1d_axis(out, axis=1)
        out = conv1d_axis(out, axis=0)
    return out


def _resize_2d_bilinear(img, new_h, new_w):
    """Resize 2D array to (new_h, new_w) using bilinear interpolation (no cv2/skimage)."""
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    if new_h == h and new_w == w:
        return img

    # Interpolate along x
    x_old = np.linspace(0.0, 1.0, w, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, new_w, dtype=np.float32)
    tmp = np.empty((h, new_w), dtype=np.float32)
    for i in range(h):
        tmp[i] = np.interp(x_new, x_old, img[i])

    # Interpolate along y
    y_old = np.linspace(0.0, 1.0, h, dtype=np.float32)
    y_new = np.linspace(0.0, 1.0, new_h, dtype=np.float32)
    out = np.empty((new_h, new_w), dtype=np.float32)
    for j in range(new_w):
        out[:, j] = np.interp(y_new, y_old, tmp[:, j])

    return out


def build_mask(image_filename, n_freq_bins, max_mask_time_bins=2048, mask_type="edges",
               edge_dilate=1, blur_passes=1, flip_freq=False, flip_time=False):
    """
    Build a spectrogram-friendly watermark mask M with shape (n_freq_bins, Wm),
    where Wm <= max_mask_time_bins. You can interpolate columns of M over time frames.
    """
    img = mpimg.imread(image_filename)
    g = _to_grayscale01(img)

    if mask_type == "edges":
        gy, gx = np.gradient(g)
        e = np.hypot(gx, gy)
        if e.max() > 1e-8:
            e = e / e.max()
        # keep strong edges for clarity
        thr = np.quantile(e, 0.80) if e.size > 0 else 0.0
        m = (e >= thr).astype(np.float32)
        m = _dilate_max(m, k=3, iters=int(edge_dilate))
    elif mask_type == "luma":
        m = g.astype(np.float32)
    else:
        raise ValueError("mask_type must be 'edges' or 'luma'")

    # Light blur to reduce clicky artifacts after embedding
    m = _blur_separable(m, passes=int(blur_passes))

    # Optional orientation controls (used for standalone mask generation).
    # For watermark embedding, flips are applied per-frame in the STFT loop for reliability.
    # - flip_freq: flip vertically (image top <-> bottom in frequency)
    # - flip_time: flip horizontally (image left <-> right in time)
    if flip_freq:
        m = np.flip(m, axis=0)
    if flip_time:
        m = np.flip(m, axis=1)

    # Limit time resolution of the mask to avoid huge memory for long audio
    h, w = m.shape
    target_w = int(min(max_mask_time_bins, max(8, w)))
    m = _resize_2d_bilinear(m, new_h=int(n_freq_bins), new_w=target_w)

    # normalize to [0,1]
    m -= m.min()
    if m.max() > 1e-8:
        m /= m.max()
    return m


def _mask_column(mask, t, n_frames):
    """Get interpolated mask column (n_freq_bins,) for frame t."""
    n_freq, w = mask.shape
    if n_frames <= 1 or w <= 1:
        return mask[:, 0]
    # Map frame index to mask x coordinate
    x = (t / (n_frames - 1)) * (w - 1)
    x0 = int(np.floor(x))
    x1 = min(x0 + 1, w - 1)
    a = float(x - x0)
    return (1.0 - a) * mask[:, x0] + a * mask[:, x1]


# -------------------------
# Watermark embedding (streaming STFT)
# -------------------------
def watermark_audio_with_image(audio_in, image_in, audio_out,
                               alpha=0.10,
                               n_fft=2048,
                               hop=512,
                               fmin=2000.0,
                               fmax=16000.0,
                               mask_type="edges",
                               max_mask_time_bins=2048,
                               edge_dilate=1,
                               blur_passes=1,
                               taper_hz=250.0,
                               flip_freq=False,
                               flip_time=False,
                               mode="overlay",
                               mix=0.5,
                               depth_db=18.0):
    """
    Embed an image-derived watermark into an existing audio file's spectrogram magnitude.

    Modes:
      - overlay:   log|S| = log|A| + alpha * band_weight * signed_mask
                   (best when you want to preserve the carrier audio)
      - blend:     log|S| = (1-mix)*log|A| + mix*(mean_log|A| + k_db * band_weight * signed_mask)
                   (more aggressive; makes the image MUCH more visible, but alters audio more)

    Key knobs:
      - alpha: watermark strength in overlay mode (natural-log domain)
      - mix: blend strength in blend mode (0..1)
      - depth_db: contrast of the embedded image in blend mode, expressed in dB swing (+/- depth_db)
      - fmin/fmax: embed only within a band to protect audio quality
    """
    rate, x = load_wav_mono(audio_in)
    x = np.asarray(x, dtype=np.float32)

    if hop <= 0 or hop > n_fft:
        raise ValueError("hop must be in [1, n_fft]")

    window = np.hanning(n_fft).astype(np.float32)
    eps = 1e-7

    # Number of frames with padding for last frame
    if x.shape[0] <= n_fft:
        n_frames = 1
    else:
        n_frames = int(np.ceil((x.shape[0] - n_fft) / hop)) + 1
    out_len = n_fft + hop * (n_frames - 1)
    pad = out_len - x.shape[0]
    if pad > 0:
        x_pad = np.pad(x, (0, pad), mode="constant")
    else:
        x_pad = x

    # Build mask with limited time resolution
    n_freq_bins = n_fft // 2 + 1
    mask = build_mask(
        image_in,
        n_freq_bins=n_freq_bins,
        max_mask_time_bins=max_mask_time_bins,
        mask_type=mask_type,
        edge_dilate=edge_dilate,
        blur_passes=blur_passes,
        flip_freq=False,
        flip_time=False,
    )

    # Band weight (with gentle taper to avoid hard edges)
    freqs = np.linspace(0.0, rate / 2.0, n_freq_bins, dtype=np.float32)

    def smoothstep(a, b, x_):
        t = np.clip((x_ - a) / max(1e-8, (b - a)), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    w = np.zeros_like(freqs)
    core = (freqs >= fmin) & (freqs <= fmax)
    w[core] = 1.0
    if taper_hz > 0:
        w *= smoothstep(fmin - taper_hz, fmin + taper_hz, freqs)
        w *= (1.0 - smoothstep(fmax - taper_hz, fmax + taper_hz, freqs))
    w = w.astype(np.float32)

    # If using blend mode, compute mean log-magnitude spectrum across time (2-pass, streaming).
    mean_logmag = None
    if mode.lower() == "blend":
        acc = np.zeros(n_freq_bins, dtype=np.float64)
        for t in range(n_frames):
            i = t * hop
            frame = x_pad[i:i + n_fft]
            if frame.shape[0] < n_fft:
                frame = np.pad(frame, (0, n_fft - frame.shape[0]), mode="constant")
            X = np.fft.rfft(frame * window, n=n_fft)
            mag = np.abs(X).astype(np.float32)
            acc += np.log(mag + eps)
        mean_logmag = (acc / max(1, n_frames)).astype(np.float32)

    # Prepare overlap-add buffers
    y = np.zeros(out_len, dtype=np.float32)
    win_sum = np.zeros(out_len, dtype=np.float32)

    # Convert dB swing to natural-log swing: log(mag) += (ln(10)/20) * dB
    k_db = (np.log(10.0) / 20.0) * float(depth_db)

    mode_l = mode.lower().strip()
    if mode_l not in ("overlay", "blend"):
        raise ValueError("mode must be 'overlay' or 'blend'")

    for t in range(n_frames):
        i = t * hop
        frame = x_pad[i:i + n_fft]
        if frame.shape[0] < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.shape[0]), mode="constant")

        X = np.fft.rfft(frame * window, n=n_fft)
        mag = np.abs(X).astype(np.float32)
        phase = X / (mag + eps)

        # Watermark mask column (orientation handled here for reliability)
        t_mask = (n_frames - 1 - t) if flip_time else t
        mcol = _mask_column(mask, t_mask, n_frames).astype(np.float32)
        if flip_freq:
            mcol = mcol[::-1]  # vertical flip (frequency axis)
        mcol = 2.0 * mcol - 1.0   # signed in [-1, 1] for high contrast

        logmag = np.log(mag + eps)

        if mode_l == "overlay":
            # Mild / moderate changes: add a signed ripple
            logmag_new = logmag + (float(alpha) * w * mcol)
        else:
            # Aggressive: pull magnitude toward an image-shaped target around the mean spectrum
            # Target is independent of the carrier's time-varying harmonics, so the image reads much better.
            target = mean_logmag + (k_db * w * mcol)
            m = float(np.clip(mix, 0.0, 1.0))
            logmag_new = (1.0 - m) * logmag + m * target

        mag_new = np.exp(logmag_new) - eps
        mag_new = np.maximum(mag_new, 0.0)

        X_new = phase * mag_new
        frame_new = np.fft.irfft(X_new, n=n_fft).astype(np.float32)

        y[i:i + n_fft] += frame_new * window
        win_sum[i:i + n_fft] += window**2

    y /= (win_sum + 1e-8)
    y = y[:x.shape[0]]

    # Prevent clipping
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0.99:
        y = y * (0.99 / peak)

    write_wav_mono(audio_out, rate, y)
    print(f"Wrote watermarked audio: {audio_out}")


# -------------------------
# Original image->audio synthesis (kept for baseline)
# -------------------------
def make_wav(image_filename):
    """Make a WAV file having a spectrogram resembling an image (baseline)."""
    image = mpimg.imread(image_filename)
    image = _to_grayscale01(image)

    # transpose to put pixels into the processing orientation (time x frequency)
    image = image.T

    # flip so the generated spectrogram keeps the same vertical orientation as the input image
    image = np.flip(image, axis=0)

    # emphasize brighter pixels
    image = image**3

    w, h = image.shape
    data = np.fft.irfft(image, h * 2, axis=1).reshape((w * h * 2))
    data -= np.average(data)
    data *= (2**15 - 1.0) / np.amax(np.abs(data))
    data = array("h", np.int16(data)).tobytes()

    output_file = wave.open(image_filename + ".wav", "wb")
    output_file.setparams((1, 2, 44100, 0, "NONE", "not compressed"))
    output_file.writeframes(data)
    output_file.close()
    print("Wrote %s.wav" % image_filename)


def _main():
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(description="Image->audio spectrogram synthesis and audio+image watermark embedding.")
    sub = p.add_subparsers(dest="cmd")

    p_img = sub.add_parser("image2wav", help="Generate a WAV whose spectrogram resembles the image (baseline).")
    p_img.add_argument("image", help="Input image file (e.g., .jpg/.png)")

    p_wm = sub.add_parser("watermark", help="Embed an image into an existing audio file's spectrogram magnitude.")
    p_wm.add_argument("--audio", required=True, help="Input carrier WAV file")
    p_wm.add_argument("--image", required=True, help="Input image file for watermark")
    p_wm.add_argument("--output", required=True, help="Output WAV file")

    p_wm.add_argument("--alpha", type=float, default=0.10, help="Watermark strength (log-magnitude).")
    p_wm.add_argument("--nfft", type=int, default=2048, help="STFT window size (n_fft).")
    p_wm.add_argument("--hop", type=int, default=512, help="Hop length (samples).")

    p_wm.add_argument("--fmin", type=float, default=2000.0, help="Min frequency (Hz) for watermark band.")
    p_wm.add_argument("--fmax", type=float, default=16000.0, help="Max frequency (Hz) for watermark band.")
    p_wm.add_argument("--taper_hz", type=float, default=250.0, help="Taper width (Hz) for band edges.")

    p_wm.add_argument("--mask", choices=["edges", "luma"], default="edges", help="Watermark mask type.")
    p_wm.add_argument("--max_mask_time_bins", type=int, default=2048,
                      help="Max time resolution of watermark mask (limits memory for long audio).")
    p_wm.add_argument("--edge_dilate", type=int, default=1, help="Dilation iterations for edge mask.")
    p_wm.add_argument("--blur_passes", type=int, default=1, help="Blur passes for mask smoothing.")
    p_wm.add_argument("--flip_freq", action="store_true",
                      help="Flip the watermark vertically (frequency axis). Use only if your image appears upside-down.")
    p_wm.add_argument("--flip_time", action="store_true",
                      help="Flip the watermark horizontally (time axis). Use only if your image appears mirrored left-right.")
    p_wm.add_argument("--mode", choices=["overlay", "blend"], default="overlay",
                      help="Embedding mode: overlay (preserve audio) or blend (more visible image, alters audio more).")
    p_wm.add_argument("--mix", type=float, default=0.5,
                      help="Blend strength for mode=blend (0..1). Higher = more image visibility, more audio change.")
    p_wm.add_argument("--depth_db", type=float, default=18.0,
                      help="Image contrast for mode=blend in dB (+/- depth_db swing around the mean spectrum).")

    args = p.parse_args()

    # Backward-compatible mode: `python spectrogram.py some_image.jpg`
    if args.cmd is None:
        import sys
        if len(sys.argv) == 2 and Path(sys.argv[1]).exists():
            make_wav(sys.argv[1])
            return
        p.print_help()
        return

    if args.cmd == "image2wav":
        make_wav(args.image)
        return

    if args.cmd == "watermark":
        watermark_audio_with_image(
            audio_in=args.audio,
            image_in=args.image,
            audio_out=args.output,
            alpha=args.alpha,
            n_fft=args.nfft,
            hop=args.hop,
            fmin=args.fmin,
            fmax=args.fmax,
            mask_type=args.mask,
            max_mask_time_bins=args.max_mask_time_bins,
            edge_dilate=args.edge_dilate,
            blur_passes=args.blur_passes,
            taper_hz=args.taper_hz,
            mode=args.mode,
            mix=args.mix,
            depth_db=args.depth_db,
            flip_freq=args.flip_freq,
            flip_time=args.flip_time,
        )
        return


if __name__ == "__main__":
    _main()
