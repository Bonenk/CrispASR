package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

public class WhisperAheads extends Structure {
    /**
     * Number of alignment heads. C declares this as `size_t` (8 bytes on
     * every 64-bit platform — Win64 LLP64, Linux/macOS LP64). JNA's
     * `NativeLong` was the original type here, but `NativeLong` follows
     * C `long` semantics (4 bytes on Win64), which is a 4-byte ABI
     * mismatch against `size_t` and corrupts the struct layout — the
     * matching `dtw_mem_size` mismatch in WhisperContextParams was
     * causing `whisper_init_from_file_with_params` to read past the
     * Java-marshalled struct and JNA to throw `Invalid memory access`
     * once the windows build matrix actually ran (issue #70 bindings-
     * java cascade). The Java `long` primitive is always 8 bytes, so
     * matches `size_t` on every 64-bit target — and bindings-java only
     * ships x64 artifacts, so 32-bit divergence isn't reachable.
     */
    public long n_heads;

    public Pointer heads;

    public WhisperAheads() {
        super();
    }

    /**
     * Create alignment heads from an array of WhisperAhead objects
     */
    public void setHeads(WhisperAhead[] aheadsArray) {
        if (aheadsArray == null) throw new IllegalArgumentException("Alignment heads required");
        if (aheadsArray.length > Integer.MAX_VALUE / 8) throw new IllegalArgumentException("Too many alignment heads");
        if (aheadsArray.length == 0) { n_heads = 0; heads = null; return; }
        for (WhisperAhead head : aheadsArray)
            if (head == null) throw new IllegalArgumentException("Null alignment head");
        Memory owned = new Memory(8L * aheadsArray.length);
        for (int i = 0; i < aheadsArray.length; i++) {
            owned.setInt(i * 8L, aheadsArray[i].n_text_layer);
            owned.setInt(i * 8L + 4, aheadsArray[i].n_head);
        }
        // Publish only after validation and copying succeed.
        n_heads = aheadsArray.length;
        heads = owned;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("n_heads", "heads");
    }

    public static class ByReference extends WhisperAheads implements Structure.ByReference {}

    public static class ByValue extends WhisperAheads implements Structure.ByValue {}
}
