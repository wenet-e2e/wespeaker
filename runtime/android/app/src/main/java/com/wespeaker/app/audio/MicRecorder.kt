package com.wespeaker.app.audio

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Captures PCM via [AudioRecord]; prefers 16 kHz mono, falls back to device rates, then [WavNormalize] to 16 kHz.
 */
class MicRecorder {

    private var audioRecord: AudioRecord? = null
    private var thread: Thread? = null
    private val running = AtomicBoolean(false)
    private val chunks = mutableListOf<ShortArray>()

    fun preferredSampleRate(): Int {
        val tryRates = intArrayOf(16000, 48000, 44100, 22050)
        for (rate in tryRates) {
            val min = AudioRecord.getMinBufferSize(
                rate,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
            )
            if (min > 0) return rate
        }
        return 44100
    }

    fun start(sampleRate: Int): Boolean {
        if (running.get()) return false
        chunks.clear()
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
        )
        if (minBuf <= 0) return false
        val record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            minBuf * 2,
        )
        if (record.state != AudioRecord.STATE_INITIALIZED) {
            record.release()
            return false
        }
        audioRecord = record
        running.set(true)
        record.startRecording()
        val bufSize = minBuf / 2
        val readBuf = ShortArray(bufSize.coerceAtLeast(256))
        thread = Thread({
            while (running.get()) {
                val n = record.read(readBuf, 0, readBuf.size)
                if (n > 0) {
                    synchronized(chunks) {
                        chunks.add(readBuf.copyOf(n))
                    }
                } else if (n < 0) {
                    break
                }
            }
        }, "wespeaker-mic").also { it.start() }
        return true
    }

    fun stop(): ShortArray {
        running.set(false)
        val rec = audioRecord
        if (rec != null) {
            try {
                rec.stop()
            } catch (_: Exception) {
            }
        }
        try {
            thread?.join(5000)
        } catch (_: InterruptedException) {
            Thread.currentThread().interrupt()
        }
        thread = null
        rec?.release()
        audioRecord = null
        val merged: List<ShortArray>
        synchronized(chunks) {
            merged = chunks.toList()
            chunks.clear()
        }
        val total = merged.sumOf { it.size }
        if (total == 0) return ShortArray(0)
        val out = ShortArray(total)
        var o = 0
        for (c in merged) {
            c.copyInto(out, o)
            o += c.size
        }
        return out
    }
}
