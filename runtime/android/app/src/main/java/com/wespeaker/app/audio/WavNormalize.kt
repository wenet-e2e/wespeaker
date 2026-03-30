package com.wespeaker.app.audio

import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 将任意常见 PCM / float WAV 转为模型所需：单声道、16 kHz、16-bit PCM，并写出标准 WAV。
 */
object WavNormalize {

    const val TARGET_SAMPLE_RATE = 16000
    const val TARGET_CHANNELS = 1

    data class PcmMono16(val samples: ShortArray, val sampleRate: Int)

    /**
     * 从 WAV 字节解析为单声道 int16 序列（若为多声道则平均；若采样率非 16k 则线性重采样）。
     */
    fun wavBytesToMono16k(bytes: ByteArray): PcmMono16 {
        val parsed = parseWav(bytes)
        var mono = toMono(parsed.samples, parsed.channels)
        mono = resampleLinear(mono, parsed.sampleRate, TARGET_SAMPLE_RATE)
        return PcmMono16(mono, TARGET_SAMPLE_RATE)
    }

    /** 已是单声道 PCM 时，仅重采样到 16 kHz（麦克风录制用）。 */
    fun monoPcmTo16k(samples: ShortArray, sampleRate: Int): ShortArray {
        if (sampleRate == TARGET_SAMPLE_RATE) return samples
        return resampleLinear(samples, sampleRate, TARGET_SAMPLE_RATE)
    }

    fun writeMono16Wav(file: File, samples: ShortArray, sampleRate: Int = TARGET_SAMPLE_RATE) {
        val dataSize = samples.size * 2
        val riffSize = 36 + dataSize
        FileOutputStream(file).use { os ->
            val hdr = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
            hdr.put("RIFF".toByteArray())
            hdr.putInt(riffSize)
            hdr.put("WAVE".toByteArray())
            hdr.put("fmt ".toByteArray())
            hdr.putInt(16)
            hdr.putShort(1) // PCM
            hdr.putShort(1) // mono
            hdr.putInt(sampleRate)
            hdr.putInt(sampleRate * 2)
            hdr.putShort(2)
            hdr.putShort(16)
            hdr.put("data".toByteArray())
            hdr.putInt(dataSize)
            os.write(hdr.array())
            val buf = ByteBuffer.allocate(samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
            for (s in samples) buf.putShort(s)
            os.write(buf.array())
        }
    }

    private data class ParsedWav(
        val samples: ShortArray,
        val channels: Int,
        val sampleRate: Int,
    )

    private fun parseWav(bytes: ByteArray): ParsedWav {
        require(bytes.size >= 44) { "WAV 过短" }
        require(String(bytes, 0, 4) == "RIFF") { "非 RIFF" }
        require(String(bytes, 8, 4) == "WAVE") { "非 WAVE" }

        var pos = 12
        var audioFormat = 0
        var numChannels = 0
        var sampleRate = 0
        var bitsPerSample = 0
        var dataOffset = -1
        var dataSize = 0

        while (pos + 8 <= bytes.size) {
            val id = String(bytes, pos, 4)
            val size = ByteBuffer.wrap(bytes, pos + 4, 4).order(ByteOrder.LITTLE_ENDIAN).int
            val contentStart = pos + 8
            if (id == "fmt ") {
                require(size >= 16) { "fmt 块无效" }
                val bb = ByteBuffer.wrap(bytes, contentStart, size).order(ByteOrder.LITTLE_ENDIAN)
                audioFormat = bb.short.toInt() and 0xffff
                numChannels = bb.short.toInt() and 0xffff
                sampleRate = bb.int
                bb.int // byte rate
                bb.short // block align
                bitsPerSample = bb.short.toInt() and 0xffff
            } else if (id == "data") {
                dataOffset = contentStart
                dataSize = size
                break
            }
            pos = contentStart + size + (size and 1)
        }
        require(dataOffset >= 0 && dataSize > 0) { "缺少 data 块" }
        require(numChannels in 1..16) { "声道数异常: $numChannels" }

        val samples: ShortArray = when (audioFormat) {
            1 -> when (bitsPerSample) {
                16 -> decodePcm16Interleaved(bytes, dataOffset, dataSize, numChannels)
                8 -> decodePcm8Interleaved(bytes, dataOffset, dataSize, numChannels)
                else -> throw IllegalArgumentException("不支持的 PCM 位深: $bitsPerSample")
            }
            3 -> {
                require(bitsPerSample == 32) { "float WAV 需 32-bit" }
                decodeFloat32Interleaved(bytes, dataOffset, dataSize, numChannels)
            }
            else -> throw IllegalArgumentException("不支持的 WAV 格式码: $audioFormat")
        }
        return ParsedWav(samples, numChannels, sampleRate)
    }

    private fun decodePcm16Interleaved(
        bytes: ByteArray,
        offset: Int,
        dataSize: Int,
        channels: Int,
    ): ShortArray {
        val total = dataSize / 2
        val out = ShortArray(total)
        val bb = ByteBuffer.wrap(bytes, offset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until total) {
            out[i] = bb.short
        }
        return out
    }

    private fun decodePcm8Interleaved(
        bytes: ByteArray,
        offset: Int,
        dataSize: Int,
        channels: Int,
    ): ShortArray {
        val total = dataSize
        val out = ShortArray(total)
        for (i in 0 until total) {
            val u = bytes[offset + i].toInt() and 0xff
            out[i] = ((u - 128) * 256).coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt()).toShort()
        }
        return out
    }

    private fun decodeFloat32Interleaved(
        bytes: ByteArray,
        offset: Int,
        dataSize: Int,
        channels: Int,
    ): ShortArray {
        val floatCount = dataSize / 4
        val out = ShortArray(floatCount)
        val bb = ByteBuffer.wrap(bytes, offset, dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until floatCount) {
            val f = bb.float
            val s = (f * 32767.0f).toInt().coerceIn(Short.MIN_VALUE.toInt(), Short.MAX_VALUE.toInt())
            out[i] = s.toShort()
        }
        return out
    }

    private fun toMono(interleaved: ShortArray, channels: Int): ShortArray {
        if (channels == 1) return interleaved
        val frames = interleaved.size / channels
        val out = ShortArray(frames)
        for (i in 0 until frames) {
            var sum = 0L
            for (c in 0 until channels) {
                sum += interleaved[i * channels + c].toInt()
            }
            out[i] = (sum / channels).toInt().toShort()
        }
        return out
    }

    private fun resampleLinear(input: ShortArray, srcRate: Int, dstRate: Int): ShortArray {
        if (srcRate == dstRate || input.isEmpty()) return input
        require(srcRate > 0 && dstRate > 0)
        val outLen = ((input.size.toLong() * dstRate + srcRate / 2) / srcRate).toInt().coerceAtLeast(1)
        val out = ShortArray(outLen)
        for (i in 0 until outLen) {
            val srcPos = (i.toDouble() * srcRate) / dstRate
            val idx = srcPos.toInt().coerceIn(0, input.size - 1)
            val frac = srcPos - idx
            val s0 = input[idx].toDouble()
            val s1 = if (idx + 1 < input.size) input[idx + 1].toDouble() else s0
            val v = (s0 + (s1 - s0) * frac).toInt().coerceIn(
                Short.MIN_VALUE.toInt(),
                Short.MAX_VALUE.toInt(),
            )
            out[i] = v.toShort()
        }
        return out
    }
}
