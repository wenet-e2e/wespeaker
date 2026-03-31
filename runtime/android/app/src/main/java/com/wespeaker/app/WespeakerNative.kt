package com.wespeaker.app

object WespeakerNative {
    init {
        System.loadLibrary("wespeaker_jni")
    }

    /**
     * @return floatArrayOf(score, sameFlag) — sameFlag 1f means same speaker (score >= threshold).
     * Embedding dim is inferred from ONNX output shape; features use the **full** utterance (no chunking).
     */
    @JvmStatic
    external fun compare(
        enrollPath: String,
        testPath: String,
        modelPath: String,
        threshold: Double,
        fbankDim: Int,
        sampleRate: Int,
    ): FloatArray
}
