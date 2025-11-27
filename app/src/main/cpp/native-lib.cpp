#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include "sentencepiece/src/sentencepiece_processor.h"

// A global pointer to the processor (Simpler than passing it around for this demo)
// In production, you might want to store this in a Java 'long' handle.
static sentencepiece::SentencePieceProcessor* processor = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_nilio_poetryhour_NativeTokenizer_loadModel(
        JNIEnv* env,
        jobject /* this */,
        jstring path) {

    const char* nativePath = env->GetStringUTFChars(path, 0);

    if (processor) {
        delete processor;
    }
    processor = new sentencepiece::SentencePieceProcessor();
    
    const auto status = processor->Load(nativePath);
    
    env->ReleaseStringUTFChars(path, nativePath);

    if (!status.ok()) {
        __android_log_print(ANDROID_LOG_ERROR, "SPM_JNI", "Failed to load: %s", status.ToString().c_str());
        return JNI_FALSE;
    }
    
    return JNI_TRUE;
}

extern "C" JNIEXPORT jintArray JNICALL
Java_com_nilio_poetryhour_NativeTokenizer_encodeNative(
        JNIEnv* env,
        jobject /* this */,
        jstring text) {

    if (!processor) return env->NewIntArray(0);

    const char* nativeText = env->GetStringUTFChars(text, 0);
    
    std::vector<int> ids;
    processor->Encode(nativeText, &ids);
    
    env->ReleaseStringUTFChars(text, nativeText);

    // Convert C++ vector to Java IntArray
    jintArray result = env->NewIntArray(ids.size());
    env->SetIntArrayRegion(result, 0, ids.size(), ids.data());

    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_nilio_poetryhour_NativeTokenizer_decodeNative(
        JNIEnv* env,
        jobject /* this */,
        jint id) {
    
    if (!processor) return env->NewStringUTF("");

    std::string piece = processor->IdToPiece(id);
    
    // SentencePiece uses a special underscore (U+2581). 
    // You might want to replace it here or in Kotlin.
    return env->NewStringUTF(piece.c_str());
}
