package com.nilio.poetryhour

import android.content.Context
import android.util.Log
import java.io.File

object NativeTokenizer {
    private var isLoaded = false

    // The special "Lower One Eighth Block" character used by SentencePiece
    private const val SPIECE_UNDERLINE = "\u2581"

    init {
        try {
            System.loadLibrary("sentencepiece_native")
        } catch (e: UnsatisfiedLinkError) {
            Log.e("Tokenizer", "CRITICAL: Could not load native library.", e)
        }
    }

    // --- JNI DECLARATIONS (Renamed to 'Native') ---
    private external fun loadModel(path: String): Boolean
    private external fun encodeNative(text: String): IntArray
    private external fun decodeNative(id: Int): String

    // --- PUBLIC WRAPPERS (Handle Spaces <-> Underscores) ---

    fun init(context: Context, assetName: String = "tokenizer.spm"): Boolean {
        if (isLoaded) return true
        val file = File(context.filesDir, assetName)
        if (!file.exists() || file.length() == 0L) {
            try {
                context.assets.open(assetName).use { input ->
                    java.io.FileOutputStream(file).use { output -> input.copyTo(output) }
                }
            } catch (e: Exception) { return false }
        }
        val success = loadModel(file.absolutePath)
        if (success) isLoaded = true
        return success
    }

    /**
     * Replaces spaces with U+2581 before sending to C++
     */
    fun encode(text: String): IntArray {
        if (!isLoaded) return IntArray(0)
        // Swap Space -> Underscore
        val processed = text.replace(" ", SPIECE_UNDERLINE)
        return encodeNative(processed)
    }

    /**
     * Helper to get Floats directly (Used by GemmaClient)
     */
    fun encodeAsFloats(text: String): FloatArray {
        val ints = encode(text)
        val floats = FloatArray(ints.size)
        for (i in ints.indices) {
            floats[i] = ints[i].toFloat()
        }
        return floats
    }

    /**
     * Replaces U+2581 with spaces after receiving from C++
     */
    fun decode(id: Int): String {
        if (!isLoaded) return ""
        val piece = decodeNative(id)
        // Swap Underscore -> Space
        return piece.replace(SPIECE_UNDERLINE, " ")
    }
}