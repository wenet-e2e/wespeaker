import numpy as np
import soundfile
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_file',
                        type=str,
                        default=None,
                        help='single wav file')
    parser.add_argument(
        '--seconds',
        type=float,
        required=False,
        default=None,
        help='how long of the audio will be used as test sample')

    FLAGS = parser.parse_args()
    wav_file = FLAGS.audio_file
    waveform, sample_rate = soundfile.read(wav_file)
    true_length = len(waveform) // sample_rate
    if FLAGS.seconds:
        num_samples = int(FLAGS.second * sample_rate)
        seconds = FLAGS.seconds
        if seconds < true_length:
            waveform = waveform[0:num_samples]
        else:
            temp = np.zeros(num_samples, dtype=np.float32)
            temp[0:len(waveform)] = waveform[:]
            waveform = temp

    data = {
        "data": [{
            "WAV": {
                "shape": [len(waveform)],
                "content": waveform.tolist()
            }
        }]
    }

    json.dump(data, open("input.json", "w"))
