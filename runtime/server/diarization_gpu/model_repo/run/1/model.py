import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import to_dlpack, from_dlpack
import torch
import numpy as np
import json
import asyncio


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance
            device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        self.model_config = model_config = json.loads(args['model_config'])
        self.max_batch_size = max(model_config["max_batch_size"], 1)

        if "GPU" in model_config["instance_group"][0]["kind"]:
            self.device = "cuda"
        else:
            self.device = "cpu"

        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "LABELS")
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

        self.init_jit_model("/workspace/triton/silero_vad.jit")

    def init_jit_model(self, model_path):
        torch.set_grad_enabled(False)
        self.sad_model = torch.jit.load(model_path, map_location=self.device)
        self.sad_model.eval()

    def prepare_chunks(self,
                       wav,
                       audio_length_samples,
                       sr: int = 16000,
                       window_size_samples: int = 1536):
        chunks = []
        self.sad_model.reset_states()

        for current_start_sample in range(0, audio_length_samples,
                                          window_size_samples):
            chunk = wav[current_start_sample:current_start_sample +
                        window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = torch.nn.functional.pad(
                    chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob = self.sad_model(chunk, 16000)
            chunks.append(speech_prob)
        return chunks

    def get_timestamps(self,
                       speech_probs,
                       audio_length_samples,
                       sr: int = 16000,
                       threshold: float = 0.5,
                       min_duration: float = 0.255,
                       min_speech_duration_ms: int = 250,
                       min_silence_duration_ms: int = 100,
                       window_size_samples: int = 1536,
                       speech_pad_ms: int = 30):
        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15
        temp_end = 0

        min_speech_samples = sr * min_speech_duration_ms / 1000
        min_silence_samples = sr * min_silence_duration_ms / 1000
        speech_pad_samples = sr * speech_pad_ms / 1000

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech['start'] = window_size_samples * i
                continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech['end'] = temp_end
                    if (current_speech['end'] -
                            current_speech['start']) > min_speech_samples:
                        speeches.append(current_speech)
                    temp_end = 0
                    current_speech = {}
                    triggered = False
                    continue
        if current_speech:
            current_speech['end'] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech['start'] = int(
                    max(0, speech['start'] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]['start'] - speech['end']
                if silence_duration < 2 * speech_pad_samples:
                    speech['end'] += int(silence_duration // 2)
                    speeches[i + 1]['start'] = int(
                        max(0,
                            speeches[i + 1]['start'] - silence_duration // 2))
                else:
                    speech['end'] += int(speech_pad_samples)
            else:
                speech['end'] = int(
                    min(audio_length_samples,
                        speech['end'] + speech_pad_samples))
        vad_result = []
        for item in speeches:
            begin = item['start'] / sr
            end = item['end'] / sr
            if end - begin >= min_duration:
                item['start'] = begin
                item['end'] = end
                vad_result.append(item)
        return vad_result

    def subsegment(self,
                   wav,
                   segments,
                   wav_idx,
                   window_fs: float = 1.50,
                   period_fs: float = 0.75,
                   sr: int = 16000,
                   frame_shift: int = 10):

        def repeat_to_fill(x, window_fs):
            length = x.size(0)
            num = (window_fs + length - 1) // length

            x = x.repeat(1, num)[0][:window_fs]
            input = torch.zeros((1, window_fs), device=self.device)
            input[0] = x
            return input

        subsegs = []
        subseg_signals = []

        seg_idx = 0

        window_fs = int(window_fs * sr)
        period_fs = int(period_fs * sr)
        for segment in segments:
            seg_begin = int(segment['start'] * sr)
            seg_end = int(segment['end'] * sr)
            seg_signal = wav[seg_begin:seg_end + 1]
            seg_length = seg_end - seg_begin

            if seg_length <= window_fs:
                subseg = [
                    wav_idx, seg_idx, segment['start'], segment['end'], 0,
                    int(seg_length / sr * 1000 // frame_shift)
                ]
                subseg_signal = repeat_to_fill(seg_signal, window_fs)

                subsegs.append(subseg)
                subseg_signals.append(subseg_signal)
                seg_idx += 1
            else:
                max_subseg_begin = seg_length - window_fs + period_fs
                for subseg_begin in range(0, max_subseg_begin, period_fs):
                    subseg_end = min(subseg_begin + window_fs, seg_length)
                    subseg = [
                        wav_idx, seg_idx, segment['start'], segment['end'],
                        int(subseg_begin / sr * 1000 / frame_shift),
                        int(subseg_end / sr * 1000 / frame_shift)
                    ]
                    subseg_signal = repeat_to_fill(
                        seg_signal[subseg_begin:subseg_end + 1], window_fs)

                    subsegs.append(subseg)
                    subseg_signals.append(subseg_signal)
                    seg_idx += 1

        return subsegs, subseg_signals

    def read_labels(self, subseg_ids, label, frame_shift=10):
        utt_to_subseg_labels = []
        new_sort = {}
        for i, subseg in enumerate(subseg_ids):
            (utt, seg_idx, begin_ms, end_ms, begin_frames, end_frames) = subseg
            begin = (int(begin_ms * 1000) +
                     int(begin_frames) * frame_shift) / 1000.0
            end = (int(begin_ms * 1000) +
                   int(end_frames) * frame_shift) / 1000.0
            new_sort[seg_idx] = (begin, end, label[i])
        utt_to_subseg_labels = list(dict(sorted(new_sort.items())).values())
        return utt_to_subseg_labels

    def merge_segments(self, subseg_to_labels):
        merged_segment_to_labels = []

        if len(subseg_to_labels) == 0:
            return merged_segment_to_labels

        (begin, end, label) = subseg_to_labels[0]
        for (b, e, la) in subseg_to_labels[1:]:
            if b <= end and la == label:
                end = e
            elif b > end:
                merged_segment_to_labels.append((begin, end, label))
                begin, end, label = b, e, la
            elif b <= end and la != label:
                pivot = (b + end) / 2.0
                merged_segment_to_labels.append((begin, pivot, label))
                begin, end, label = pivot, e, la
            else:
                raise ValueError
        merged_segment_to_labels.append((begin, e, label))

        return merged_segment_to_labels

    async def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        batch_count = []
        batch_len = []

        total_wavs = []
        total_lens = []
        responses = []

        for request in requests:
            input0 = pb_utils.get_input_tensor_by_name(request, "input")

            cur_b_wav = from_dlpack(input0.to_dlpack())
            cur_batch = cur_b_wav.shape[0]
            cur_len = cur_b_wav.shape[1]
            batch_count.append(cur_batch)
            batch_len.append(cur_len)

            for wav in cur_b_wav:
                total_lens.append(len(wav))
                total_wavs.append(wav.to(self.device))

        speech_shapes = []
        all_probs = []

        for wav, lens in zip(total_wavs, total_lens):
            chunks = self.prepare_chunks(wav, lens)
            speech_shapes.append(len(chunks))
            all_probs.append(chunks)
        reshape_probs = []
        idx = 0
        for i in range(0, len(speech_shapes)):
            cur_speech = []
            for j in range(0, speech_shapes[i]):
                cur_speech.append(all_probs[i][j])
                idx += 1
            reshape_probs.append(cur_speech)

        out_segs = []
        for speech_prob, speech_len in zip(reshape_probs, total_lens):
            segments = self.get_timestamps(speech_prob,
                                           speech_len,
                                           threshold=0.36)
            out_segs.append(segments)

        total_subsegments = []
        total_subsegment_ids = []
        total_embds = []

        wav_idx = 0
        for waveform, segments in zip(total_wavs, out_segs):
            subsegs, subseg_signals = self.subsegment(waveform, segments,
                                                      wav_idx)
            total_subsegments.extend(subseg_signals)
            total_subsegment_ids.extend(subsegs)
            wav_idx += 1

        inference_response_awaits = []
        for wavs in total_subsegments:
            input_tensor_spk0 = pb_utils.Tensor.from_dlpack(
                "WAV", to_dlpack(wavs))

            input_tensors_spk = [input_tensor_spk0]
            inference_request = pb_utils.InferenceRequest(
                model_name='speaker',
                requested_output_names=['EMBEDDINGS'],
                inputs=input_tensors_spk)
            inference_response_awaits.append(inference_request.async_exec())

        inference_responses = await asyncio.gather(*inference_response_awaits)

        for inference_response in inference_responses:
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message())
            else:
                batched_result = pb_utils.get_output_tensor_by_name(
                    inference_response, 'EMBEDDINGS')
                total_embds.extend(from_dlpack(batched_result.to_dlpack()))

        out_embds = list()
        out_time_info = list()
        for i in range(0, len(total_wavs)):
            out_embds.append(list())
            out_time_info.append(list())

        for subseg_idx, embds in zip(total_subsegment_ids, total_embds):
            wav_idx = subseg_idx[0]
            out_embds[wav_idx].append(embds)
            out_time_info[wav_idx].append(subseg_idx)

        # Begin clustering
        inference_response_awaits = []
        for i, embd in enumerate(out_embds):
            embd = torch.stack(embd)
            input_tensor_embds0 = pb_utils.Tensor.from_dlpack(
                "EMBEDDINGS", to_dlpack(torch.unsqueeze(embd, 0)))

            input_tensors_spk = [input_tensor_embds0]
            inference_request = pb_utils.InferenceRequest(
                model_name='clusterer',
                requested_output_names=['LABELS'],
                request_id=str(i),
                inputs=input_tensors_spk)
            inference_response_awaits.append(inference_request.async_exec())

        inference_responses = await asyncio.gather(*inference_response_awaits)

        i = 0
        results = []
        for inference_response in inference_responses:
            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message())
            else:
                result = pb_utils.get_output_tensor_by_name(
                    inference_response, 'LABELS').as_numpy()[0]
                utt_to_subseg_labels = self.read_labels(
                    out_time_info[i], result)
                i += 1
                rttm = self.merge_segments(utt_to_subseg_labels)
                if len(rttm) > 0:
                    results.append(rttm)

        # Return the batched resoponse
        st = 0
        for b in batch_count:
            sents = np.array(results[st:st + b])
            out0 = pb_utils.Tensor("LABELS", sents.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out0])
            responses.append(inference_response)
            st += b
        return responses
