import com.sun.jna.*;
import io.github.ggerganov.whispercpp.*;
import io.github.ggerganov.whispercpp.params.*;
import java.util.concurrent.atomic.AtomicInteger;
public class JavaLegacyProbe {
  public interface Probe extends Library {
    int probe_legacy_live();
    int probe_legacy_contexts();
    int probe_legacy_dtw();
    void probe_legacy_mode(int n);
    int probe_legacy_gpu_device();
    int probe_legacy_use_gpu();
    int probe_legacy_flash();
    int probe_legacy_head_count();
    int probe_legacy_head_value();
    long probe_legacy_dtw_memory();
    int probe_legacy_full_size();
  }
  static int count;
  interface Action {
    void run() throws Exception;
  }
  static void eq(Object a, Object b) {
    if (!java.util.Objects.equals(a, b))
      throw new AssertionError(a + " != " + b);
    ++count;
  }
  static void rejects(Class<? extends Throwable> type, Action action) throws Exception {
    try {
      action.run();
    } catch (Throwable error) {
      if (type.isInstance(error)) {
        ++count;
        return;
      }
      throw new AssertionError(error);
    }
    throw new AssertionError("Missing " + type);
  }
  public static void main(String[] args) throws Exception {
    Probe p = Native.load("crispasr", Probe.class);
    WhisperCpp model = new WhisperCpp();
    WhisperContextParams.ByValue options = model.getContextDefaultParams();
    eq(p.probe_legacy_live(), 0);
    options.enableDtwTokenTimestamps(true);
    options.setDtwAheadsPreset(2);
    options.dtw_n_top = 2;
    options.gpu_device = 3;
    options.useGpu(false);
    options.useFlashAttn(false);
    options.dtw_mem_size = 123456;
    WhisperAhead head = new WhisperAhead();
    head.n_text_layer = 2;
    head.n_head = 3;
    options.dtw_aheads.setHeads(new WhisperAhead[] {head});
    model.initContext("/model", options);
    eq(p.probe_legacy_dtw(), 202);
    eq(p.probe_legacy_gpu_device(), 3);
    eq(p.probe_legacy_use_gpu(), 0);
    eq(p.probe_legacy_flash(), 0);
    eq(p.probe_legacy_dtw_memory(), 123456L);
    eq(p.probe_legacy_head_count(), 1);
    eq(p.probe_legacy_head_value(), 203);
    options.dtw_aheads.heads.setInt(0, 99);
    options.dtw_aheads.setHeads(new WhisperAhead[0]);
    System.gc();
    eq(p.probe_legacy_head_value(), 203);
    eq(p.probe_legacy_contexts(), 1);
    eq(p.probe_legacy_live(), 0);
    WhisperFullParams.ByValue full =
        model.getFullDefaultParams(WhisperSamplingStrategy.CRISPASR_SAMPLING_GREEDY);
    eq(full.size(), p.probe_legacy_full_size());
    full.language = "de";
    full.initial_prompt = "Grüße";
    eq(p.probe_legacy_live(), 0);
    eq(model.fullTranscribe(full, new float[] {1, 3, 5}), "日");
    eq(model.fullTranscribeWithTime(full, new float[] {1, 3, 5}).size(), 1);
    AtomicInteger calls = new AtomicInteger();
    full.setProgressCallback((ctx, state, progress, data) -> {
      if (progress != 37)
        throw new AssertionError();
      try {
        rejects(IllegalStateException.class, model::close);
        rejects(IllegalStateException.class, () -> model.initContext("/reentry", options));
      } catch (Exception error) {
        throw new AssertionError(error);
      }
      calls.incrementAndGet();
    });
    System.gc();
    eq(model.fullTranscribe(full, new float[] {1, 3, 5}), "日");
    eq(calls.get(), 1);
    p.probe_legacy_mode(6);
    eq(model.fullTranscribe(full, new float[] {1, 3, 5}), "日");
    eq(calls.get(), 2);
    p.probe_legacy_mode(0);
    full.setProgressCallback(null);
    eq(model.fullTranscribe(full, new float[] {1, 3, 5}), "日");
    eq(calls.get(), 2);
    p.probe_legacy_mode(1);
    eq(model.fullTranscribe(full, new float[] {1, 3, 5}), "");
    eq(model.fullTranscribeWithTime(full, new float[] {1, 3, 5}).size(), 0);
    p.probe_legacy_mode(7);
    rejects(java.io.IOException.class, () -> model.fullTranscribe(full, new float[] {1, 3, 5}));
    rejects(java.io.IOException.class, () -> model.fullTranscribeWithTime(full, new float[] {1, 3, 5}));
    p.probe_legacy_mode(0);
    rejects(IllegalArgumentException.class, () -> model.fullTranscribe(full, new float[0]));
    p.probe_legacy_mode(2);
    rejects(java.io.IOException.class, () -> model.fullTranscribe(full, new float[] {1, 3, 5}));
    p.probe_legacy_mode(0);
    p.probe_legacy_mode(3);
    rejects(IllegalStateException.class,
        () -> model.getFullDefaultParams(WhisperSamplingStrategy.CRISPASR_SAMPLING_GREEDY));
    p.probe_legacy_mode(0);
    p.probe_legacy_mode(4);
    rejects(IllegalStateException.class, () -> model.initContext("/model", options));
    p.probe_legacy_mode(0);
    eq(p.probe_legacy_contexts(), 1);
    rejects(IllegalArgumentException.class, () -> model.initContext("bad\0path", options));
    eq(p.probe_legacy_contexts(), 1);
    rejects(java.io.FileNotFoundException.class, () -> model.initContext("/fail", options));
    eq(p.probe_legacy_contexts(), 0);
    model.close();
    eq(p.probe_legacy_contexts(), 0);
    rejects(IllegalStateException.class, () -> model.fullTranscribe(full, new float[] {1, 3, 5}));
    model.initContext("/model", options);
    model.initContext("/model", options);
    eq(p.probe_legacy_contexts(), 1);
    model.close();
    model.close();
    eq(p.probe_legacy_contexts(), 0);
    eq(p.probe_legacy_live(), 0);
    eq(full.language, "de");
    System.out.println("PASS Java legacy: " + count + " assertions");
  }
}
