package com.nilio.poetryhour

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.PriorityQueue
import kotlin.math.exp
import kotlin.math.min
import kotlin.random.Random

class GemmaClient(context: Context, modelName: String) {

    private var interpreter: Interpreter? = null

    // Buffers & Maps
    private val inputBuffers = HashMap<Int, ByteBuffer>()
    private val outputBuffers = HashMap<Int, ByteBuffer>()
    private var previousInput:String = ""
    var step = 0
    private var vocabSize = 262144
    private val CACHE_MAPPING = listOf(
        Pair(40, 38), // Layer k_0
        Pair(17, 17), // Layer v_0
        Pair(46, 44), // Layer k_1
        Pair(35, 35), // Layer v_1
        Pair(1, 1), // Layer k_2
        Pair(45, 43), // Layer v_2
        Pair(16, 16), // Layer k_3
        Pair(0, 0), // Layer v_3
        Pair(12, 12), // Layer k_4
        Pair(49, 47), // Layer v_4
        Pair(52, 50), // Layer k_5
        Pair(21, 21), // Layer v_5
        Pair(22, 22), // Layer k_6
        Pair(10, 10), // Layer v_6
        Pair(25, 25), // Layer k_7
        Pair(33, 33), // Layer v_7
        Pair(3, 3), // Layer k_8
        Pair(47, 45), // Layer v_8
        Pair(14, 14), // Layer k_9
        Pair(44, 42), // Layer v_9
        Pair(7, 6), // Layer k_10
        Pair(26, 26), // Layer v_10
        Pair(50, 48), // Layer k_11
        Pair(20, 20), // Layer v_11
        Pair(18, 18), // Layer k_12
        Pair(28, 28), // Layer v_12
        Pair(32, 32), // Layer k_13
        Pair(39, 37), // Layer v_13
        Pair(30, 30), // Layer k_14
        Pair(54, 52), // Layer v_14
        Pair(43, 41), // Layer k_15
        Pair(31, 31), // Layer v_15
        Pair(2, 2), // Layer k_16
        Pair(27, 27), // Layer v_16
        Pair(9, 9), // Layer k_17
        Pair(38, 36), // Layer v_17
        Pair(23, 23), // Layer k_18
        Pair(19, 19), // Layer v_18
        Pair(51, 49), // Layer k_19
        Pair(24, 24), // Layer v_19
        Pair(13, 13), // Layer k_20
        Pair(53, 51), // Layer v_20
        Pair(41, 39), // Layer k_21
        Pair(34, 34), // Layer v_21
        Pair(11, 11), // Layer k_22
        Pair(6, 5), // Layer v_22
        Pair(29, 29), // Layer k_23
        Pair(15, 15), // Layer v_23
        Pair(8, 7), // Layer k_24
        Pair(42, 40), // Layer v_24
        Pair(48, 46), // Layer k_25
        Pair(5, 4), // Layer v_25
    )

    // Indices
    private val IDX_TOKEN = 36
    private val IDX_POS = 4
    private val IDX_MASK = 37
    private var IDX_LOGITS = -1

    init {
        val options = Interpreter.Options()

        Log.i("GemmaClient", "Loadding mapped file")
        val modelFile = FileUtil.loadMappedFile(context, "models/$modelName")
        Log.i("GemmaClient", "Loading Interpreter")
        interpreter = Interpreter(modelFile, options)
        Log.i("GemmaClient", "Done loading Interpreter")
        val session = interpreter!!
        session.allocateTensors()

        val inputNames =
            (0 until session.inputTensorCount).associateWith {
                session.getInputTensor(it).name()
            }
        val outputNames =
            (0 until session.outputTensorCount).associateWith {
                session.getOutputTensor(it).name()
            }

        Log.i("GemmaClient", "Inputs: $inputNames")
        Log.i("GemmaClient", "Outputs: $outputNames")

        // Setup Inputs/Outputs
        Log.i("GemmaClient", "Input tensor count: ${session.inputTensorCount}")
        for (i in 0 until session.inputTensorCount) {
            val tensor = session.getInputTensor(i)
            inputBuffers[i] =
                ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
        }
        Log.i("GemmaClient", "Output tensor count: ${session.outputTensorCount}")
        for (i in 0 until session.outputTensorCount) {
            val tensor = session.getOutputTensor(i)
            outputBuffers[i] =
                ByteBuffer.allocateDirect(tensor.numBytes()).order(ByteOrder.nativeOrder())
            if (tensor.shape().last() > 32000) {
                IDX_LOGITS = i
                vocabSize = tensor.shape().last()
            }
        }
    }

    fun format(text: String): String {
        return "<start_of_turn>user\nWrite a poem.<end_of_turn><start_of_turn>model\n$text"
    }

    fun predictNextPossibleWords(text: String, topK: Int = 5, temperature: Float = 1.0f): List<String> {
        val session = interpreter ?: return emptyList()

        var tokens: IntArray = intArrayOf()
        if (previousInput != "" && text.startsWith(previousInput)) {
            val prompt = text.slice(previousInput.length until text.length)
            tokens = NativeTokenizer.encode(prompt)
            previousInput += prompt
        } else {
            clearCache()
            val prompt = format(text)
            val rawTokens = NativeTokenizer.encode(prompt)
            tokens = if (rawTokens.firstOrNull() != 2) intArrayOf(2) + rawTokens else rawTokens
            previousInput = text
            step = 0
        }

        var lastLogits: ByteBuffer? = null
        inputBuffers[IDX_MASK]!!.clear()
        val maskFloats = inputBuffers[IDX_MASK]!!.asFloatBuffer()
        for (i in 0 until 2048) {
            maskFloats.put(-1e9f)
        }

        // --- 1. INFERENCE LOOP ---
        for (token in tokens) {
            // 1. Token Input (Index 36)
            inputBuffers[IDX_TOKEN]?.clear()
            inputBuffers[IDX_TOKEN]?.asIntBuffer()?.put(token)

            // 2. Position Input (Index 4)
            inputBuffers[IDX_POS]?.clear()
            inputBuffers[IDX_POS]?.asIntBuffer()?.put(step)

            // 3. Attention Mask (Index 37)
            // The mask is FLOAT32. We must fill the entire context window (2048).
            // 1.0  = Valid (Pay attention to this position)
            // -1e9 = Invalid (Ignore/Mask out this position)
            val maskBuffer = inputBuffers[IDX_MASK]!!
            maskBuffer.clear()
            val maskFloats = maskBuffer.asFloatBuffer()

            for (i in 0 until step) {
                maskFloats.put(1f)
            }
            maskBuffer.rewind()

            // 4. Run Inference
            val inputsArray = arrayOfNulls<Any>(session.inputTensorCount)
            inputBuffers.forEach { (k, v) -> v.rewind(); inputsArray[k] = v }

            val outputsMap = HashMap<Int, Any>()
            outputBuffers.forEach { (k, v) -> v.clear(); outputsMap[k] = v }

            session.runForMultipleInputsOutputs(inputsArray, outputsMap)

            // 5. Rotate Cache (Output -> Input)
            for ((inIdx, outIdx) in CACHE_MAPPING) {
                safeCopy(outputBuffers[outIdx]!!, inputBuffers[inIdx]!!)
            }

            step += 1
        }

        lastLogits = outputBuffers[IDX_LOGITS]
        if (lastLogits == null) return emptyList()

        val rawLogits = FloatArray(vocabSize)
        lastLogits.rewind()
        lastLogits.asFloatBuffer().get(rawLogits)

        val candidates = getTopCandidates(rawLogits, 200)


        val softmaxOp = SoftmaxWithTemperatureOp(temperature)
        val probabilities = softmaxOp.apply(candidates)

        val suggestions = ArrayList<String>()
        val pool = ArrayList(probabilities)
        for (i in 0 until topK) {
            if (pool.isEmpty()) break

            while (true) {
                // Spin the roulette wheel
                val choiceIndex = sampleIndex(pool)
                val choice = pool[choiceIndex]

                // Remove picked item so we don't suggest it twice
                pool.removeAt(choiceIndex)

                // Add valid words only
                if (isEnglish(choice.word)) {
                    suggestions.add(choice.word)
                    break
                } else {
                    // If we picked a bad word, try one more time in this slot
                    // (Optional logic to keep UI consistent)
                    if (pool.isEmpty()) break
                }
            }
        }

        return suggestions
    }

    // --- HELPERS ---

    data class Candidate(val id: Int, val word: String, var logit: Float)

    // Get the raw top N scores efficiently
    private fun getTopCandidates(logits: FloatArray, k: Int): List<Candidate> {
        val pq = PriorityQueue<Candidate>(k) { a, b -> a.logit.compareTo(b.logit) } // Min Heap

        for (i in logits.indices) {
            val score = logits[i]
            if (score > Float.NEGATIVE_INFINITY) {
                if (pq.size < k) {
                    pq.add(Candidate(i, NativeTokenizer.decode(i), score))
                } else if (score > pq.peek().logit) {
                    pq.poll()
                    pq.add(Candidate(i, NativeTokenizer.decode(i), score))
                }
            }
        }
        return pq.sortedByDescending { it.logit }
    }

    // Spin the roulette wheel based on logit
    private fun sampleIndex(candidates: List<Candidate>): Int {
        val target = Random.nextFloat()
        var cumulative = 0f
        for ((i, cand) in candidates.withIndex()) {
            cumulative += cand.logit
            if (cumulative >= target) return i
        }
        return candidates.lastIndex // Fallback (floating point errors)
    }

    private fun isEnglish(text: String): Boolean {
        // Allow words, apostrophes, and basic punctuation
        return text.isNotEmpty() && text.matches(Regex("^[\\s'a-zA-Z,.:;\\-]+$"))
    }

    private fun clearCache() {
        val controls = listOf(IDX_TOKEN, IDX_POS, IDX_MASK)
        for ((idx, buffer) in inputBuffers) {
            if (controls.contains(idx)) continue
            buffer.clear()
            while(buffer.hasRemaining()) buffer.put(0.toByte())
            buffer.clear()
        }
    }

    private fun safeCopy(source: ByteBuffer, dest: ByteBuffer) {
        source.rewind()
        dest.clear()
        val limit = min(source.capacity(), dest.capacity())
        source.limit(limit)
        dest.put(source)
        source.limit(source.capacity())
    }

    /**
     * Custom Operator to apply Temperature Scaling and Softmax.
     * We perform this on the reduced set of candidates for performance.
     */
    private class SoftmaxWithTemperatureOp(private val temperature: Float) {

        fun apply(candidates: List<Candidate>): List<Candidate> {
            val maxLogit = candidates.maxByOrNull { it.logit }?.logit ?: 0f

            var sum = 0.0
            for (cand in candidates) {
                val adjusted = (cand.logit - maxLogit) / temperature
                val p = exp(adjusted.toDouble())
                cand.logit = p.toFloat() // Store un-normalized probability
                sum += p
            }

            for (cand in candidates) {
                cand.logit = (cand.logit / sum).toFloat()
            }

            return candidates
        }
    }
}

