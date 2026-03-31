package com.wespeaker.app

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.wespeaker.app.audio.MicRecorder
import com.wespeaker.app.audio.WavNormalize
import com.wespeaker.app.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private var enrollFile: File? = null
    private var testFile: File? = null

    private val micRecorder = MicRecorder()
    /** 0=enroll, 1=test, null=not recording */
    private var recordingSlot: Int? = null
    private var captureSampleRate: Int = WavNormalize.TARGET_SAMPLE_RATE

    private var pendingAfterMicPermission: (() -> Unit)? = null

    private val requestMicPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            val run = pendingAfterMicPermission
            pendingAfterMicPermission = null
            if (granted) {
                run?.invoke()
            } else {
                Toast.makeText(this, "需要麦克风权限才能录制", Toast.LENGTH_SHORT).show()
            }
        }

    private val pickEnroll =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                lifecycleScope.launch {
                    prepareWavFromUri(uri, isEnroll = true)
                }
            }
        }

    private val pickTest =
        registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
            if (uri != null) {
                lifecycleScope.launch {
                    prepareWavFromUri(uri, isEnroll = false)
                }
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnEnroll.setOnClickListener {
            pickEnroll.launch(arrayOf("audio/*", "audio/wav", "audio/x-wav"))
        }
        binding.btnTest.setOnClickListener {
            pickTest.launch(arrayOf("audio/*", "audio/wav", "audio/x-wav"))
        }
        binding.btnEnrollRecord.setOnClickListener { toggleMicRecording(isEnroll = true) }
        binding.btnTestRecord.setOnClickListener { toggleMicRecording(isEnroll = false) }
        binding.btnCompare.setOnClickListener { runCompare() }
    }

    private suspend fun prepareWavFromUri(uri: android.net.Uri, isEnroll: Boolean) {
        val tv = if (isEnroll) binding.tvEnroll else binding.tvTest
        try {
            tv.text = getString(R.string.processing_audio)
            val bytes = withContext(Dispatchers.IO) {
                contentResolver.openInputStream(uri)!!.use { it.readBytes() }
            }
            val pcm = WavNormalize.wavBytesToMono16k(bytes)
            val out = File(cacheDir, if (isEnroll) "enroll.wav" else "test.wav")
            withContext(Dispatchers.IO) {
                WavNormalize.writeMono16Wav(out, pcm.samples)
            }
            if (isEnroll) {
                enrollFile = out
            } else {
                testFile = out
            }
            tv.text = getString(
                R.string.label_audio_ready,
                out.name,
                pcm.samples.size,
            )
        } catch (e: Exception) {
            if (isEnroll) enrollFile = null else testFile = null
            tv.text = getString(R.string.audio_prepare_failed, e.message ?: e.javaClass.simpleName)
        }
    }

    private fun toggleMicRecording(isEnroll: Boolean) {
        val wantSlot = if (isEnroll) 0 else 1
        if (recordingSlot != null && recordingSlot != wantSlot) {
            Toast.makeText(this, R.string.stop_other_recording_first, Toast.LENGTH_SHORT).show()
            return
        }
        if (recordingSlot == wantSlot) {
            stopMicAndSave(isEnroll)
            return
        }
        ensureMicPermissionThen {
            val sr = micRecorder.preferredSampleRate()
            captureSampleRate = sr
            if (!micRecorder.start(sr)) {
                Toast.makeText(this, R.string.mic_open_failed, Toast.LENGTH_SHORT).show()
                return@ensureMicPermissionThen
            }
            recordingSlot = wantSlot
            refreshRecordingUi()
        }
    }

    private fun ensureMicPermissionThen(block: () -> Unit) {
        when {
            ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED -> block()
            else -> {
                pendingAfterMicPermission = block
                requestMicPermission.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }

    private fun stopMicAndSave(isEnroll: Boolean) {
        lifecycleScope.launch(Dispatchers.Default) {
            val sr = captureSampleRate
            val samples = micRecorder.stop()
            recordingSlot = null
            withContext(Dispatchers.Main) { refreshRecordingUi() }
            if (samples.isEmpty()) {
                withContext(Dispatchers.Main) {
                    Toast.makeText(this@MainActivity, "未采集到有效音频", Toast.LENGTH_SHORT).show()
                }
                return@launch
            }
            val mono16k = WavNormalize.monoPcmTo16k(samples, sr)
            val out = File(cacheDir, if (isEnroll) "enroll.wav" else "test.wav")
            withContext(Dispatchers.IO) {
                WavNormalize.writeMono16Wav(out, mono16k)
            }
            if (isEnroll) {
                enrollFile = out
            } else {
                testFile = out
            }
            val tv = if (isEnroll) binding.tvEnroll else binding.tvTest
            withContext(Dispatchers.Main) {
                tv.text = getString(
                    R.string.label_audio_ready,
                    out.name,
                    mono16k.size,
                )
            }
        }
    }

    private fun refreshRecordingUi() {
        val enrollRec = recordingSlot == 0
        val testRec = recordingSlot == 1
        binding.btnEnrollRecord.text = if (enrollRec) {
            getString(R.string.stop_recording)
        } else {
            getString(R.string.record_enroll)
        }
        binding.btnTestRecord.text = if (testRec) {
            getString(R.string.stop_recording)
        } else {
            getString(R.string.record_test)
        }
        binding.btnEnroll.isEnabled = recordingSlot == null
        binding.btnTest.isEnabled = recordingSlot == null
        binding.btnEnrollRecord.isEnabled = !testRec
        binding.btnTestRecord.isEnabled = !enrollRec
        binding.btnCompare.isEnabled = recordingSlot == null
        if (enrollRec) {
            binding.tvEnroll.text = getString(R.string.recording_hint_enroll)
        }
        if (testRec) {
            binding.tvTest.text = getString(R.string.recording_hint_test)
        }
    }

    private fun runCompare() {
        val e = enrollFile
        val t = testFile
        if (e == null || !e.exists() || t == null || !t.exists()) {
            Toast.makeText(this, "请先选择或录制两段音频", Toast.LENGTH_SHORT).show()
            return
        }
        val threshold = binding.etThreshold.text?.toString()?.toDoubleOrNull() ?: 0.5

        binding.btnCompare.isEnabled = false

        lifecycleScope.launch {
            try {
                val modelFile = withContext(Dispatchers.IO) { copyModelFromAssetsIfPresent() }
                if (modelFile == null) {
                    showCompareDialog(
                        getString(R.string.dialog_title_tip),
                        getString(R.string.model_missing),
                    )
                    return@launch
                }
                val out = withContext(Dispatchers.Default) {
                    WespeakerNative.compare(
                        e.absolutePath,
                        t.absolutePath,
                        modelFile.absolutePath,
                        threshold,
                        80,
                        16000,
                    )
                }
                if (out == null || out.size < 2) {
                    showCompareDialog(
                        getString(R.string.dialog_title_tip),
                        "推理失败（返回为空）",
                    )
                    return@launch
                }
                val score = out[0]
                val same = out[1] >= 0.5f
                val verdict =
                    getString(if (same) R.string.verdict_same else R.string.verdict_diff)
                showCompareDialog(
                    getString(R.string.dialog_title_compare_result),
                    getString(
                        R.string.dialog_msg_score_format,
                        "%.4f".format(score),
                        verdict,
                    ),
                )
            } catch (ex: Exception) {
                showCompareDialog(
                    getString(R.string.dialog_title_error),
                    ex.message ?: ex.javaClass.simpleName,
                )
            } finally {
                binding.btnCompare.isEnabled = true
            }
        }
    }

    private fun showCompareDialog(title: String, message: String) {
        MaterialAlertDialogBuilder(this)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton(R.string.dialog_positive_ok, null)
            .show()
    }

    /** If assets contain final.onnx, copy to app files dir for native loading */
    private fun copyModelFromAssetsIfPresent(): File? {
        val out = File(filesDir, "final.onnx")
        if (out.exists() && out.length() > 0) return out
        return try {
            assets.open(ASSET_ONNX).use { input ->
                java.io.FileOutputStream(out).use { output -> input.copyTo(output) }
            }
            out
        } catch (_: Exception) {
            null
        }
    }

    companion object {
        private const val ASSET_ONNX = "final.onnx"
    }
}
