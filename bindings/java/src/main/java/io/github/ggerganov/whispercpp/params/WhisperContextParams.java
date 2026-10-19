package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

/**
 * Parameters for the whisper_init_from_file_with_params() function.
 * If you change the order or add new parameters, make sure to update the default values in crispasr:
 * whisper_context_default_params()
 */
public class WhisperContextParams extends Structure {
    public WhisperContextParams(Pointer p) {
        super(p);
    }

    public WhisperContextParams() {
        super();
    }

    /** Use GPU for inference (default = true) */
    public CBool use_gpu;

    /** Use flash attention (default = true) */
    public CBool flash_attn;

    /** CUDA device to use (default = 0) */
    public int gpu_device;

    /** [EXPERIMENTAL] Enable token-level timestamps with DTW (default = false) */
    public CBool dtw_token_timestamps;

    /** [EXPERIMENTAL] Alignment heads preset for DTW */
    public int dtw_aheads_preset;

    /** Number of top layers to use for DTW when using CRISPASR_AHEADS_N_TOP_MOST preset */
    public int dtw_n_top;

    public WhisperAheads.ByValue dtw_aheads;

    /**
     * DTW memory size (internal use). C declares this as `size_t` —
     * 8 bytes on every 64-bit platform — so use the Java `long`
     * primitive (always 8 bytes), not `NativeLong` (4 bytes on Win64).
     * Mismatch caused `whisper_init_from_file_with_params` to read
     * past the Java-marshalled struct by 4 bytes once bindings-java
     * actually started running on Windows after the C1061 fix landed,
     * surfacing as a JNA `Invalid memory access` error. See WhisperAheads.n_heads.
     */
    public long dtw_mem_size;

    /** Use GPU for inference */
    public void useGpu(boolean enable) {
        use_gpu = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Use flash attention */
    public void useFlashAttn(boolean enable) {
        flash_attn = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Enable DTW token-level timestamps */
    public void enableDtwTokenTimestamps(boolean enable) {
        dtw_token_timestamps = enable ? CBool.TRUE : CBool.FALSE;
    }

    /** Set DTW alignment heads preset */
    public void setDtwAheadsPreset(int preset) {
        dtw_aheads_preset = preset;
    }

    /** Pointer-width-aware native layout; the public fields remain Java configuration values. */
    public static class SizeT extends IntegerType {
        public SizeT() {
            this(0);
        }
        public SizeT(long value) {
            super(Native.SIZE_T_SIZE, value, true);
        }
    }
    public static class NativeAheads extends Structure {
        public SizeT n_heads = new SizeT();
        public Pointer heads;
        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("n_heads", "heads");
        }
    }
    public static class NativeLayout extends Structure {
        public CBool use_gpu, flash_attn;
        public int gpu_device;
        public CBool dtw_token_timestamps;
        public int dtw_aheads_preset, dtw_n_top;
        public NativeAheads dtw_aheads;
        public SizeT dtw_mem_size;
        public NativeLayout(Pointer pointer) {
            super(pointer);
        }
        @Override
        protected List<String> getFieldOrder() {
            return Arrays.asList("use_gpu", "flash_attn", "gpu_device", "dtw_token_timestamps",
                    "dtw_aheads_preset", "dtw_n_top", "dtw_aheads", "dtw_mem_size");
        }
    }
    public static ByValue readNative(Pointer pointer) {
        NativeLayout nativeParams = new NativeLayout(pointer);
        nativeParams.read();
        ByValue out = new ByValue();
        out.use_gpu = nativeParams.use_gpu;
        out.flash_attn = nativeParams.flash_attn;
        out.gpu_device = nativeParams.gpu_device;
        out.dtw_token_timestamps = nativeParams.dtw_token_timestamps;
        out.dtw_aheads_preset = nativeParams.dtw_aheads_preset;
        out.dtw_n_top = nativeParams.dtw_n_top;
        out.dtw_aheads = new WhisperAheads.ByValue();
        out.dtw_aheads.n_heads = nativeParams.dtw_aheads.n_heads.longValue();
        out.dtw_aheads.heads = nativeParams.dtw_aheads.heads;
        out.dtw_mem_size = nativeParams.dtw_mem_size.longValue();
        return out.snapshot();
    }
    /** Retain an independent copy of custom alignment heads for the model lifetime. */
    public ByValue snapshot() {
        ByValue out = new ByValue();
        out.use_gpu = new CBool(use_gpu == null ? 1 : use_gpu.longValue());
        out.flash_attn = new CBool(flash_attn == null ? 1 : flash_attn.longValue());
        out.gpu_device = gpu_device;
        out.dtw_token_timestamps =
                new CBool(dtw_token_timestamps == null ? 0 : dtw_token_timestamps.longValue());
        out.dtw_aheads_preset = dtw_aheads_preset;
        out.dtw_n_top = dtw_n_top;
        if (dtw_mem_size < 0 || (Native.SIZE_T_SIZE == 4 && dtw_mem_size > 0xffffffffL))
            throw new IllegalArgumentException("Invalid DTW memory size");
        out.dtw_mem_size = dtw_mem_size;
        out.dtw_aheads = new WhisperAheads.ByValue();
        long count = dtw_aheads == null ? 0 : dtw_aheads.n_heads;
        if (count < 0 || count > Integer.MAX_VALUE / 8)
            throw new IllegalArgumentException("Invalid alignment-head count");
        out.dtw_aheads.n_heads = count;
        if (count > 0) {
            if (dtw_aheads.heads == null)
                throw new IllegalArgumentException("Missing alignment heads");
            byte[] bytes = dtw_aheads.heads.getByteArray(0, (int) (count * 8));
            Memory owned = new Memory(bytes.length);
            owned.write(0, bytes, 0, bytes.length);
            out.dtw_aheads.heads = owned;
        }
        return out;
    }
    public void writeNative(Pointer pointer) {
        NativeLayout out = new NativeLayout(pointer);
        out.use_gpu = use_gpu;
        out.flash_attn = flash_attn;
        out.gpu_device = gpu_device;
        out.dtw_token_timestamps = dtw_token_timestamps;
        out.dtw_aheads_preset = dtw_aheads_preset;
        out.dtw_n_top = dtw_n_top;
        out.dtw_aheads.n_heads = new SizeT(dtw_aheads.n_heads);
        out.dtw_aheads.heads = dtw_aheads.heads;
        out.dtw_mem_size = new SizeT(dtw_mem_size);
        out.write();
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("use_gpu", "flash_attn", "gpu_device", "dtw_token_timestamps",
                "dtw_aheads_preset", "dtw_n_top", "dtw_aheads", "dtw_mem_size");
    }

    public static class ByValue extends WhisperContextParams implements Structure.ByValue {
        public ByValue() {
            super();
        }
        public ByValue(Pointer p) {
            super(p);
        }
    }
}
