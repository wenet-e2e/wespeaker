package com.wespeaker.app

object WespeakerNative {
    init {
        System.loadLibrary("wespeaker_jni")
    }

    /**
     * @return floatArrayOf(score, sameFlag) — sameFlag 为 1f 表示同一人（>= 阈值）
     * embedding 维度由 ONNX 输出 shape 推断；特征按**整段**提取（不分块）。
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
