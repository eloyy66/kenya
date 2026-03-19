@file:Suppress("MemberVisibilityCanBePrivate")

package com.example.mostacho

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

class DriverDistractionClassifier(
    context: Context,
    modelAssetName: String,
    private val threshold: Float = 0.72f,
    private val useNnapiIfAvailable: Boolean = true,
) {
    data class Prediction(
        val distractedProbability: Float,
        val predictedLabel: Int, // 0 normal, 1 distracted
        val labelName: String,
    )

    private val nnapiDelegate: NnApiDelegate? = if (useNnapiIfAvailable) runCatching { NnApiDelegate() }.getOrNull() else null
    private val interpreter: Interpreter
    private val inputShape: IntArray
    private val inputDataType: DataType
    private val outputDataType: DataType
    private val inputScale: Float
    private val inputZeroPoint: Int
    private val outputScale: Float
    private val outputZeroPoint: Int

    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            nnapiDelegate?.let { addDelegate(it) }
        }
        interpreter = Interpreter(loadModelFile(context, modelAssetName), options)
        val inputTensor = interpreter.getInputTensor(0)
        val outputTensor = interpreter.getOutputTensor(0)
        inputShape = inputTensor.shape() // [1, H, W, 3]
        inputDataType = inputTensor.dataType()
        outputDataType = outputTensor.dataType()
        inputScale = inputTensor.quantizationParams().scale
        inputZeroPoint = inputTensor.quantizationParams().zeroPoint
        outputScale = outputTensor.quantizationParams().scale
        outputZeroPoint = outputTensor.quantizationParams().zeroPoint
    }

    fun close() {
        interpreter.close()
        nnapiDelegate?.close()
    }

    fun classify(bitmap: Bitmap): Prediction {
        val resized = Bitmap.createScaledBitmap(bitmap, inputShape[2], inputShape[1], true)
        val inputBuffer = when (inputDataType) {
            DataType.FLOAT32 -> bitmapToFloatBuffer(resized)
            DataType.INT8, DataType.UINT8 -> bitmapToQuantizedBuffer(resized, inputDataType, inputScale, inputZeroPoint)
            else -> throw IllegalStateException("Unsupported input dtype: $inputDataType")
        }

        val outputBuffer = when (outputDataType) {
            DataType.FLOAT32 -> ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
            DataType.INT8, DataType.UINT8 -> ByteBuffer.allocateDirect(1).order(ByteOrder.nativeOrder())
            else -> throw IllegalStateException("Unsupported output dtype: $outputDataType")
        }

        interpreter.run(inputBuffer, outputBuffer)

        val probability = when (outputDataType) {
            DataType.FLOAT32 -> {
                outputBuffer.rewind()
                outputBuffer.float
            }
            DataType.INT8 -> {
                outputBuffer.rewind()
                val q = outputBuffer.get().toInt()
                ((q - outputZeroPoint) * outputScale).coerceIn(0f, 1f)
            }
            DataType.UINT8 -> {
                outputBuffer.rewind()
                val q = outputBuffer.get().toInt() and 0xFF
                ((q - outputZeroPoint) * outputScale).coerceIn(0f, 1f)
            }
            else -> throw IllegalStateException("Unsupported output dtype: $outputDataType")
        }

        val predicted = if (probability >= threshold) 1 else 0
        return Prediction(
            distractedProbability = probability,
            predictedLabel = predicted,
            labelName = if (predicted == 1) "distracted" else "normal",
        )
    }

    private fun bitmapToFloatBuffer(bitmap: Bitmap): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val buffer = ByteBuffer.allocateDirect(width * height * 3 * 4).order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        var idx = 0
        while (idx < pixels.size) {
            val px = pixels[idx]
            val r = (px shr 16 and 0xFF).toFloat()
            val g = (px shr 8 and 0xFF).toFloat()
            val b = (px and 0xFF).toFloat()
            // Modelo entrenado con entrada 0..255 y Rescaling interna a [-1,1].
            buffer.putFloat(r)
            buffer.putFloat(g)
            buffer.putFloat(b)
            idx++
        }
        buffer.rewind()
        return buffer
    }

    private fun bitmapToQuantizedBuffer(
        bitmap: Bitmap,
        dtype: DataType,
        scale: Float,
        zeroPoint: Int,
    ): ByteBuffer {
        val width = bitmap.width
        val height = bitmap.height
        val buffer = ByteBuffer.allocateDirect(width * height * 3).order(ByteOrder.nativeOrder())
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (px in pixels) {
            val rgb = intArrayOf(px shr 16 and 0xFF, px shr 8 and 0xFF, px and 0xFF)
            for (channel in rgb) {
                val q = (channel / scale + zeroPoint).roundToInt()
                when (dtype) {
                    DataType.INT8 -> buffer.put(q.coerceIn(-128, 127).toByte())
                    DataType.UINT8 -> buffer.put((q.coerceIn(0, 255) and 0xFF).toByte())
                    else -> error("Unexpected dtype: $dtype")
                }
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun loadModelFile(context: Context, modelAssetName: String): ByteBuffer {
        context.assets.openFd(modelAssetName).use { fileDescriptor ->
            FileInputStream(fileDescriptor.fileDescriptor).channel.use { fileChannel ->
                return fileChannel.map(
                    FileChannel.MapMode.READ_ONLY,
                    fileDescriptor.startOffset,
                    fileDescriptor.declaredLength,
                )
            }
        }
    }
}

