package io.github.ggerganov.whispercpp;

import com.sun.jna.Native;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.bean.WhisperSegment;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Before calling most methods, you must call `initContext(modelPath)` to initialise the `ctx` Pointer.
 */
public class WhisperCpp implements AutoCloseable {
    private WhisperCppJnaLibrary lib = WhisperCppJnaLibrary.instance;
    private Pointer ctx = null;
    private WhisperContextParams.ByValue contextParamsOwner;
    private boolean operation;
    private synchronized void beginOperation() {
        if (operation) throw new IllegalStateException("Whisper context is in use");
        operation = true;
    }
    private synchronized void endOperation() { operation = false; }

    public File modelDir() {
        String modelDirPath = System.getenv("XDG_CACHE_HOME");
        if (modelDirPath == null) {
            modelDirPath = System.getProperty("user.home") + "/.cache";
        }

        return new File(modelDirPath, "whisper");
    }

    /**
     * @param modelPath - absolute path, or just the name (eg: "base", "base-en" or "base.en")
     */
    public void initContext(String modelPath) throws FileNotFoundException {
        initContextImpl(modelPath, getContextDefaultParams());
    }

    /**
     * @param modelPath - absolute path, or just the name (eg: "base", "base-en" or "base.en")
     * @param params - params to use when initialising the context
     */
    public void initContext(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        initContextImpl(modelPath, params);
    }

    private void initContextImpl(String modelPath, WhisperContextParams.ByValue params) throws FileNotFoundException {
        if (modelPath == null || modelPath.isEmpty() || modelPath.indexOf('\0') >= 0)
            throw new IllegalArgumentException("Nonempty model path without NUL required");
        if (params == null) throw new IllegalArgumentException("Context parameters required");
        beginOperation();
        try {
            WhisperContextParams.ByValue snapshot = params.snapshot();
            if (!modelPath.contains("/") && !modelPath.contains("\\")) {
                if (!modelPath.endsWith(".bin")) modelPath = "ggml-" + modelPath.replace("-", ".") + ".bin";
                modelPath = new File(modelDir(), modelPath).getAbsolutePath();
            }
            Pointer nativeParams = lib.whisper_context_default_params_by_ref();
            if (nativeParams == null) throw new IllegalStateException("Context parameter allocation failed");
            try {
                snapshot.writeNative(nativeParams);
                freeContext();
                ctx = lib.whisper_init_from_file_with_params_by_ref(modelPath, nativeParams);
                if (ctx == null) throw new FileNotFoundException(modelPath);
                contextParamsOwner = snapshot;
            } finally {
                lib.whisper_free_context_params(nativeParams);
            }
        } finally { endOperation(); }
    }

    /** Return an independent Java-owned copy of native context defaults. */
    public WhisperContextParams.ByValue getContextDefaultParams() {
        Pointer pointer = lib.whisper_context_default_params_by_ref();
        if (pointer == null) throw new IllegalStateException("Context parameter allocation failed");
        try { return WhisperContextParams.readNative(pointer); }
        finally { lib.whisper_free_context_params(pointer); }
    }

    /** Return Java-owned decoding defaults; never free these with Native.free(). */
    public WhisperFullParams.ByValue getFullDefaultParams(WhisperSamplingStrategy strategy) {
        // The public legacy full-params structure contains 64-bit size_t fields.
        // Generic CrispasrSession is the supported route on 32-bit Java runtimes.
        if (Native.SIZE_T_SIZE != 8) throw new UnsupportedOperationException("Legacy full parameters require a 64-bit runtime");
        if (strategy == null) throw new IllegalArgumentException("Sampling strategy required");
        Pointer pointer = lib.whisper_full_default_params_by_ref(strategy.ordinal());
        if (pointer == null) throw new IllegalStateException("Decoding parameter allocation failed");
        try {
            WhisperFullParams.ByValue out = new WhisperFullParams.ByValue();
            out.getPointer().write(0, pointer.getByteArray(0, out.size()), 0, out.size());
            out.read();
            return out;
        } finally { lib.whisper_free_params(pointer); }
    }

    @Override
    public void close() {
        beginOperation();
        try { freeContext(); }
        finally { endOperation(); }
    }

    private void freeContext() {
        Pointer old = ctx;
        ctx = null;
        try { if (old != null) lib.whisper_free(old); }
        finally {
            java.lang.ref.Reference.reachabilityFence(contextParamsOwner);
            contextParamsOwner = null;
        }
    }

    /**
     * Run the entire model: PCM -&gt; log mel spectrogram -&gt; encoder -&gt; decoder -&gt; text.
     * Overlap and callback re-entry on the same context are rejected.
     * Do not mutate the parameter object or its borrowed pointer fields during inference.
     * Uses the specified decoding strategy to obtain the text.
     */
    public String fullTranscribe(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        beginOperation();
        try {
            if (ctx == null) {
                throw new IllegalStateException("Model not initialised");
            }

            if (whisperParams == null || audioData == null || audioData.length == 0)
                throw new IllegalArgumentException("Parameters and nonempty PCM required");
            if (Native.SIZE_T_SIZE != 8) throw new UnsupportedOperationException("Legacy full parameters require a 64-bit runtime");
            whisperParams.write();
            if (lib.whisper_full_by_ref(ctx, whisperParams.getPointer(), audioData, audioData.length) != 0) {
                throw new IOException("Failed to process audio");
            }

            int nSegments = lib.whisper_full_n_segments(ctx);
            if (nSegments < 0) throw new IOException("Invalid native segment count");

            StringBuilder str = new StringBuilder();

            for (int i = 0; i < nSegments; i++) {
                String text = lib.whisper_full_get_segment_text(ctx, i);
                str.append(text);
            }

            return str.toString().trim();
        } finally {
            java.lang.ref.Reference.reachabilityFence(whisperParams);
            java.lang.ref.Reference.reachabilityFence(contextParamsOwner);
            endOperation();
        }
    }

    /**
     * Full transcribe with time list.
     *
     * @param whisperParams the whisper params
     * @param audioData     the audio data
     * @return the list
     * @throws IOException the io exception
     */
    public List<WhisperSegment> fullTranscribeWithTime(WhisperFullParams.ByValue whisperParams, float[] audioData) throws IOException {
        beginOperation();
        try {
            if (ctx == null) {
                throw new IllegalStateException("Model not initialised");
            }

            if (whisperParams == null || audioData == null || audioData.length == 0)
                throw new IllegalArgumentException("Parameters and nonempty PCM required");
            if (Native.SIZE_T_SIZE != 8) throw new UnsupportedOperationException("Legacy full parameters require a 64-bit runtime");
            whisperParams.write();
            if (lib.whisper_full_by_ref(ctx, whisperParams.getPointer(), audioData, audioData.length) != 0) {
                throw new IOException("Failed to process audio");
            }

            int nSegments = lib.whisper_full_n_segments(ctx);
            if (nSegments < 0) throw new IOException("Invalid native segment count");
            List<WhisperSegment> segments= new ArrayList<>(nSegments);

            for (int i = 0; i < nSegments; i++) {
                long t0 = lib.whisper_full_get_segment_t0(ctx, i);
                String text = lib.whisper_full_get_segment_text(ctx, i);
                long t1 = lib.whisper_full_get_segment_t1(ctx, i);
                segments.add(new WhisperSegment(t0,t1,text));
            }

            return segments;
        } finally {
            java.lang.ref.Reference.reachabilityFence(whisperParams);
            java.lang.ref.Reference.reachabilityFence(contextParamsOwner);
            endOperation();
        }
    }

//    public int getTextSegmentCount(Pointer ctx) {
//        return lib.whisper_full_n_segments(ctx);
//    }
//    public String getTextSegment(Pointer ctx, int index) {
//        return lib.whisper_full_get_segment_text(ctx, index);
//    }

    public String getSystemInfo() {
        return lib.whisper_print_system_info();
    }

    public int benchMemcpy(int nthread) {
        return lib.whisper_bench_memcpy(nthread);
    }

    public int benchGgmlMulMat(int nthread) {
        return lib.whisper_bench_ggml_mul_mat(nthread);
    }
}
