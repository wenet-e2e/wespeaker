# calculate filterbank features.
# Provides e.g. fbank and mfcc features for use in ASR applications
# Author: James Lyons 2012
# GitHub: https://github.com/ZitengWang/python_kaldi_features.git
from __future__ import division
import numpy
import decimal
import logging
from scipy.fftpack import dct


def mfcc(signal,
         samplerate=16000,
         winlen=0.025,
         winstep=0.01,
         numcep=13,
         nfilt=23,
         nfft=512,
         lowfreq=20,
         highfreq=None,
         dither=1.0,
         remove_dc_offset=True,
         preemph=0.97,
         ceplifter=22,
         useEnergy=True,
         wintype='povey'):
    """Compute MFCC features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array  # noqa
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)  # noqa
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)  # noqa
    :param numcep: the number of cepstrum to return, default 13
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2  # noqa
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.  # noqa
    :param ceplifter: apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.  # noqa
    :param appendEnergy: if this is true, the zeroth cepstral coefficient is replaced with the log of the total frame energy.  # noqa
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming  # noqa
    :returns: A numpy array of size (NUMFRAMES by numcep) containing features. Each row holds 1 feature vector.  # noqa
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, dither, remove_dc_offset, preemph,
                         wintype)
    feat = numpy.log(feat)
    feat = dct(feat, type=2, axis=1, norm='ortho')[:, :numcep]
    feat = lifter(feat, ceplifter)
    if useEnergy:
        feat[:, 0] = numpy.log(
            energy
        )  # replace first cepstral coefficient with log of frame energy
    return feat


def fbank(signal,
          samplerate=16000,
          winlen=0.025,
          winstep=0.01,
          nfilt=40,
          nfft=512,
          lowfreq=0,
          highfreq=None,
          dither=1.0,
          remove_dc_offset=True,
          preemph=0.97,
          wintype='hamming'):
    """Compute Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array  # noqa
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)  # noqa
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)  # noqa
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2  # noqa
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.  # noqa
    :param winfunc: the analysis window to apply to each frame. By default no window is applied. You can use numpy window functions here e.g. winfunc=numpy.hamming  # noqa
     winfunc=lambda x:numpy.ones((x,))
    :returns: 2 values. The first is a numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector. The  # noqa
        second return value is the energy in each frame (total energy, unwindowed)  # noqa
    """
    highfreq = highfreq or samplerate / 2
    frames, raw_frames = framesig(signal, winlen * samplerate,
                                  winstep * samplerate, dither, preemph,
                                  remove_dc_offset, wintype)
    pspec = powspec(frames, nfft)  # nearly the same until this part
    energy = numpy.sum(raw_frames**2,
                       1)  # this stores the raw energy in each frame
    energy = numpy.where(energy == 0,
                         numpy.finfo(float).eps,
                         energy)  # if energy is zero, we get problems with log

    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = numpy.dot(pspec, fb.T)  # compute the filterbank energies
    feat = numpy.where(feat == 0,
                       numpy.finfo(float).eps,
                       feat)  # if feat is zero, we get problems with log

    return feat, energy


def logfbank(signal,
             samplerate=16000,
             winlen=0.025,
             winstep=0.01,
             nfilt=40,
             nfft=512,
             lowfreq=64,
             highfreq=None,
             dither=1.0,
             remove_dc_offset=True,
             preemph=0.97,
             wintype='hamming'):
    """Compute log Mel-filterbank energy features from an audio signal.

    :param signal: the audio signal from which to compute features. Should be an N*1 array  # noqa
    :param samplerate: the samplerate of the signal we are working with.
    :param winlen: the length of the analysis window in seconds. Default is 0.025s (25 milliseconds)  # noqa
    :param winstep: the step between successive windows in seconds. Default is 0.01s (10 milliseconds)  # noqa
    :param nfilt: the number of filters in the filterbank, default 26.
    :param nfft: the FFT size. Default is 512.
    :param lowfreq: lowest band edge of mel filters. In Hz, default is 0.
    :param highfreq: highest band edge of mel filters. In Hz, default is samplerate/2  # noqa
    :param preemph: apply preemphasis filter with preemph as coefficient. 0 is no filter. Default is 0.97.  # noqa
    :returns: A numpy array of size (NUMFRAMES by nfilt) containing features. Each row holds 1 feature vector.  # noqa
    """
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft,
                         lowfreq, highfreq, dither, remove_dc_offset, preemph,
                         wintype)
    return numpy.log(feat)


def hz2mel(hz):
    """Convert a value in Hertz to Mels

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.  # noqa
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.  # noqa
    """
    return 1127 * numpy.log(1 + hz / 700.0)


def mel2hz(mel):
    """Convert a value in Mels to Hertz

    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.  # noqa
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.  # noqa
    """
    return 700 * (numpy.exp(mel / 1127.0) - 1)


def get_filterbanks(nfilt=26,
                    nfft=512,
                    samplerate=16000,
                    lowfreq=0,
                    highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond  # noqa
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)  # noqa

    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.  # noqa
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.  # noqa
    """
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)

    # check kaldi/src/feat/Mel-computations.h
    fbank = numpy.zeros([nfilt, nfft // 2 + 1])
    mel_freq_delta = (highmel - lowmel) / (nfilt + 1)
    for j in range(0, nfilt):
        leftmel = lowmel + j * mel_freq_delta
        centermel = lowmel + (j + 1) * mel_freq_delta
        rightmel = lowmel + (j + 2) * mel_freq_delta
        for i in range(0, nfft // 2):
            mel = hz2mel(i * samplerate / nfft)
            if mel > leftmel and mel < rightmel:
                if mel < centermel:
                    fbank[j, i] = (mel - leftmel) / (centermel - leftmel)
                else:
                    fbank[j, i] = (rightmel - mel) / (rightmel - centermel)
    return fbank


def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the  # noqa
    magnitude of the high frequency DCT coeffs.

    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.  # noqa
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.  # noqa
    """
    if L > 0:
        nframes, ncoeff = numpy.shape(cepstra)
        n = numpy.arange(ncoeff)
        lift = 1 + (L / 2.) * numpy.sin(numpy.pi * n / L)
        return lift * cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra


def delta(feat, N):
    """Compute delta features from a feature vector sequence.

    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.  # noqa
    :param N: For each frame, calculate delta features based on preceding and following N frames  # noqa
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.  # noqa
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N + 1)])
    delta_feat = numpy.empty_like(feat)
    padded = numpy.pad(feat, ((N, N), (0, 0)),
                       mode='edge')  # padded version of feat
    for t in range(NUMFRAMES):
        delta_feat[t] = numpy.dot(
            numpy.arange(-N, N + 1),
            padded[t:t + 2 * N +
                   1]) / denominator  # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
    return delta_feat


def round_half_up(number):
    return int(
        decimal.Decimal(number).quantize(decimal.Decimal('1'),
                                         rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1], )
    return numpy.lib.stride_tricks.as_strided(a, shape=shape,
                                              strides=strides)[::step]


def framesig(sig,
             frame_len,
             frame_step,
             dither=1.0,
             preemph=0.97,
             remove_dc_offset=True,
             wintype='hamming',
             stride_trick=True):
    """Frame a signal into overlapping frames.

    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.  # noqa
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.  # noqa
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster  # noqa
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + ((slen - frame_len) // frame_step)

    # check kaldi/src/feat/feature-window.h
    padsignal = sig[:(numframes - 1) * frame_step + frame_len]
    if wintype == 'povey':
        win = numpy.empty(frame_len)
        for i in range(frame_len):
            win[i] = (0.5 - 0.5 * numpy.cos(2 * numpy.pi /
                                            (frame_len - 1) * i))**0.85
    else:  # the hamming window
        win = numpy.hamming(frame_len)

    if stride_trick:
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(
            0, frame_len), (numframes, 1)) + numpy.tile(
                numpy.arange(0, numframes * frame_step, frame_step),
                (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(win, (numframes, 1))

    frames = frames.astype(numpy.float32)
    raw_frames = numpy.zeros(frames.shape)
    for frm in range(frames.shape[0]):
        frames[frm, :] = do_dither(frames[frm, :], dither)  # dither
        frames[frm, :] = do_remove_dc_offset(
            frames[frm, :])  # remove dc offset
        raw_frames[frm, :] = frames[frm, :]
        frames[frm, :] = do_preemphasis(frames[frm, :],
                                        preemph)  # preemphasize

    return frames * win, raw_frames


def deframesig(frames,
               siglen,
               frame_len,
               frame_step,
               winfunc=lambda x: numpy.ones((x, ))):
    """Does overlap-add procedure to undo the action of framesig.

    :param frames: the array of frames.
    :param siglen: the length of the desired signal, use 0 if unknown. Output will be truncated to siglen samples.  # noqa
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.  # noqa
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.  # noqa
    :returns: a 1-D signal.
    """
    frame_len = round_half_up(frame_len)
    frame_step = round_half_up(frame_step)
    numframes = numpy.shape(frames)[0]
    assert numpy.shape(
        frames
    )[1] == frame_len, '"frames" matrix is wrong size, 2nd dim is not equal to frame_len'  # noqa

    indices = numpy.tile(numpy.arange(
        0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step),
            (frame_len, 1)).T
    indices = numpy.array(indices, dtype=numpy.int32)
    padlen = (numframes - 1) * frame_step + frame_len

    if siglen <= 0:
        siglen = padlen

    rec_signal = numpy.zeros((padlen, ))
    window_correction = numpy.zeros((padlen, ))
    win = winfunc(frame_len)

    for i in range(0, numframes):
        window_correction[indices[i, :]] = window_correction[indices[
            i, :]] + win + 1e-15  # add a little bit so it is never zero
        rec_signal[indices[i, :]] = rec_signal[indices[i, :]] + frames[i, :]

    rec_signal = rec_signal / window_correction
    return rec_signal[0:siglen]


def magspec(frames, NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).  # noqa

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.  # noqa
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.  # noqa
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',  # noqa
            numpy.shape(frames)[1], NFFT)
    complex_spec = numpy.fft.rfft(frames, NFFT)
    return numpy.absolute(complex_spec)


def powspec(frames, NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).  # noqa

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.  # noqa
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.  # noqa
    """
    return numpy.square(magspec(frames, NFFT))


def logpowspec(frames, NFFT, norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).  # noqa

    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.  # noqa
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.  # noqa
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.  # noqa
    """
    ps = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * numpy.log10(ps)
    if norm:
        return lps - numpy.max(lps)
    else:
        return lps


def do_dither(signal, dither_value=1.0):
    signal += numpy.random.normal(size=signal.shape) * dither_value
    return signal


def do_remove_dc_offset(signal):
    signal -= numpy.mean(signal)
    return signal


def do_preemphasis(signal, coeff=0.97):
    """perform preemphasis on the input signal.

    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append((1 - coeff) * signal[0],
                        signal[1:] - coeff * signal[:-1])
