package io.github.ggerganov.whispercpp;

import com.sun.jna.*;
import com.sun.jna.ptr.*;
import java.nio.charset.StandardCharsets;
import java.util.*;

/** Shared native audio, container, metadata and text-processing utilities. */
public abstract class CrispasrUtilities {
  protected abstract Pointer nativeHandle();
  public static final class SizeT extends IntegerType {
    public SizeT() {
      this(0);
    }
    public SizeT(long value) {
      super(Native.SIZE_T_SIZE, value, true);
    }
  }
  interface Utilities extends Library {
    Utilities INSTANCE = Native.load("crispasr", Utilities.class, CrispasrSession.nativeOptions());
    byte crispasr_kokoro_lang_is_german_abi(String language);
    byte crispasr_kokoro_lang_has_native_voice_abi(String language);
    String crispasr_c_api_version();
    String crispasr_dart_helpers_version();
    void crispasr_set_gpu_backend(String backend);
    int crispasr_detect_backends_from_gguf(String path, byte[] out, int cap);
    int crispasr_backend_caps_abi(String backend, byte[] out, int cap);
    int crispasr_backend_caps_list_abi(byte[] out, int cap);
    String crispasr_session_backend(Pointer s);
    int crispasr_watermark_load_model(String path);
    int crispasr_audio_load(String path, PointerByReference out, IntByReference count, IntByReference rate);
    int crispasr_audio_load_at_rate(
        String path, int target, PointerByReference out, IntByReference count, IntByReference rate);
    int crispasr_audio_load_stereo(String path, PointerByReference left, PointerByReference right,
        IntByReference count, IntByReference rate, IntByReference channels);
    int crispasr_audio_resample(
        float[] pcm, int n, int source, int target, PointerByReference out, IntByReference count);
    void crispasr_audio_free(Pointer p);
    Pointer crispasr_pcm_to_wav(float[] pcm, int n, int rate, Pointer size);
    Pointer crispasr_c2pa_sign(
        byte[] data, SizeT size, String format, String cert, String key, Pointer outSize);
    void crispasr_c2pa_free(Pointer p);
    Pointer crispasr_punc_init(String path);
    Pointer crispasr_punc_process(Pointer ctx, String text);
    void crispasr_punc_free_text(Pointer text);
    void crispasr_punc_free(Pointer ctx);
    Pointer crispasr_truecase_init(String path);
    Pointer crispasr_truecase_process(Pointer ctx, String text);
    void crispasr_truecase_free_text(Pointer text);
    void crispasr_truecase_free(Pointer ctx);
    Pointer crispasr_pcs_init(String path);
    Pointer crispasr_pcs_process(Pointer ctx, String text);
    void crispasr_pcs_free_text(Pointer text);
    void crispasr_pcs_free(Pointer ctx);
    int crispasr_session_score_texts(Pointer session, float[] pcm, int n, String language, String prompt,
        String[] texts, int count, float[] scores, int[] tokens);
  }
  private static void status(int rc) {
    if (rc < 0)
      throw new IllegalStateException("native utility failed (rc=" + rc + ")");
  }
  private static float[] floats(Pointer p, int n) {
    if (n < 0 || (n > 0 && p == null))
      throw new IllegalStateException("invalid native audio buffer");
    return n == 0 ? new float[0] : p.getFloatArray(0, n);
  }
  private static String text(byte[] bytes) {
    int n = 0;
    while (n < bytes.length && bytes[n] != 0) n++;
    return new String(bytes, 0, n, StandardCharsets.UTF_8);
  }
  public static boolean kokoroLanguageIsGerman(String language) {
    return Utilities.INSTANCE.crispasr_kokoro_lang_is_german_abi(Objects.requireNonNull(language)) != 0;
  }
  public static boolean kokoroLanguageHasNativeVoice(String language) {
    return Utilities.INSTANCE.crispasr_kokoro_lang_has_native_voice_abi(Objects.requireNonNull(language))
        != 0;
  }
  public static String cApiVersion() {
    return Utilities.INSTANCE.crispasr_c_api_version();
  }
  public static String helperVersion() {
    return Utilities.INSTANCE.crispasr_dart_helpers_version();
  }
  public static void setGpuBackend(String name) {
    Utilities.INSTANCE.crispasr_set_gpu_backend(Objects.requireNonNull(name));
  }
  public String backend() {
    return Utilities.INSTANCE.crispasr_session_backend(nativeHandle());
  }
  public static void loadWatermarkModel(String path) {
    status(Utilities.INSTANCE.crispasr_watermark_load_model(Objects.requireNonNull(path)));
  }
  public static String[] detectBackends(String path) {
    byte[] out = new byte[65536];
    status(
        Utilities.INSTANCE.crispasr_detect_backends_from_gguf(Objects.requireNonNull(path), out, out.length));
    String value = text(out);
    return value.isEmpty() ? new String[0] : value.split("\n");
  }
  public static String[] backendCapabilities(String backend) {
    byte[] out = new byte[65536];
    status(Utilities.INSTANCE.crispasr_backend_caps_abi(Objects.requireNonNull(backend), out, out.length));
    String value = text(out);
    return value.isEmpty() ? new String[0] : value.split(",");
  }
  public static String allBackendCapabilities() {
    int required = Utilities.INSTANCE.crispasr_backend_caps_list_abi(null, 0);
    if (required == Integer.MIN_VALUE)
      throw new IllegalStateException("invalid capability buffer size");
    byte[] out = new byte[Math.max(1, Math.abs(required))];
    status(Utilities.INSTANCE.crispasr_backend_caps_list_abi(out, out.length));
    return text(out);
  }
  public static final class Audio {
    /** Mono samples or interleaved stereo; native source channel count is separate. */
    public final float[] samples;
    public final int sampleRate, channels, sourceChannels;
    Audio(float[] pcm, int rate, int channels, int sourceChannels) {
      this.samples = pcm;
      this.sampleRate = rate;
      this.channels = channels;
      this.sourceChannels = sourceChannels;
    }
  }
  public static Audio loadAudio(String path) {
    return loadAudio(path, 16000);
  }
  public static Audio loadAudio(String path, int targetRate) {
    Objects.requireNonNull(path);
    if (targetRate <= 0)
      throw new IllegalArgumentException("positive rate required");
    PointerByReference out = new PointerByReference();
    IntByReference n = new IntByReference(), rate = new IntByReference();
    try {
      int rc = targetRate == 16000
          ? Utilities.INSTANCE.crispasr_audio_load(path, out, n, rate)
          : Utilities.INSTANCE.crispasr_audio_load_at_rate(path, targetRate, out, n, rate);
      status(rc);
      return new Audio(floats(out.getValue(), n.getValue()), rate.getValue(), 1, 1);
    } finally {
      if (out.getValue() != null)
        Utilities.INSTANCE.crispasr_audio_free(out.getValue());
    }
  }
  public static Audio loadAudioStereo(String path) {
    PointerByReference left = new PointerByReference(), right = new PointerByReference();
    IntByReference n = new IntByReference(), rate = new IntByReference(), channels = new IntByReference();
    try {
      status(Utilities.INSTANCE.crispasr_audio_load_stereo(
          Objects.requireNonNull(path), left, right, n, rate, channels));
      float[] l = floats(left.getValue(), n.getValue()), r = floats(right.getValue(), n.getValue()),
              pcm = new float[Math.multiplyExact(n.getValue(), 2)];
      for (int i = 0; i < l.length; i++) {
        pcm[i * 2] = l[i];
        pcm[i * 2 + 1] = r[i];
      }
      return new Audio(pcm, rate.getValue(), 2, channels.getValue());
    } finally {
      Pointer l = left.getValue(), r = right.getValue();
      if (l != null)
        Utilities.INSTANCE.crispasr_audio_free(l);
      if (r != null && !r.equals(l))
        Utilities.INSTANCE.crispasr_audio_free(r);
    }
  }
  public static float[] resample(float[] pcm, int sourceRate, int targetRate) {
    Objects.requireNonNull(pcm);
    if (sourceRate <= 0 || targetRate <= 0)
      throw new IllegalArgumentException("positive rates required");
    PointerByReference out = new PointerByReference();
    IntByReference n = new IntByReference();
    try {
      status(Utilities.INSTANCE.crispasr_audio_resample(pcm, pcm.length, sourceRate, targetRate, out, n));
      return floats(out.getValue(), n.getValue());
    } finally {
      if (out.getValue() != null)
        Utilities.INSTANCE.crispasr_audio_free(out.getValue());
    }
  }
  private static byte[] container(Pointer p, Memory size) {
    try {
      long n = Native.SIZE_T_SIZE == 8 ? size.getLong(0) : Integer.toUnsignedLong(size.getInt(0));
      if (p == null || n < 0 || n > Integer.MAX_VALUE)
        throw new IllegalStateException("native container unavailable or too large");
      return p.getByteArray(0, (int) n);
    } finally {
      if (p != null)
        Utilities.INSTANCE.crispasr_c2pa_free(p);
    }
  }
  public static byte[] pcmToWav(float[] pcm, int rate) {
    Objects.requireNonNull(pcm);
    if (rate <= 0)
      throw new IllegalArgumentException("positive rate required");
    try (Memory size = new Memory(Native.SIZE_T_SIZE)) {
      size.clear();
      return container(Utilities.INSTANCE.crispasr_pcm_to_wav(pcm, pcm.length, rate, size), size);
    }
  }
  public static byte[] signC2pa(byte[] data, String format, String cert, String key) {
    Objects.requireNonNull(data);
    Objects.requireNonNull(format);
    if ((cert == null) != (key == null))
      throw new IllegalArgumentException("certificate and key must be paired");
    try (Memory size = new Memory(Native.SIZE_T_SIZE)) {
      size.clear();
      return container(
          Utilities.INSTANCE.crispasr_c2pa_sign(data, new SizeT(data.length), format, cert, key, size), size);
    }
  }
  public enum TextProcessorKind { TRUECASE, PCS, PUNCTUATION }
  public static final class TextProcessor implements AutoCloseable {
    private Pointer handle;
    private final TextProcessorKind kind;
    public TextProcessor(String model, boolean punctuationCapitalizationSegmentation) {
      this(model, punctuationCapitalizationSegmentation ? TextProcessorKind.PCS : TextProcessorKind.TRUECASE);
    }
    public TextProcessor(String model, TextProcessorKind kind) {
      this.kind = Objects.requireNonNull(kind, "kind");
      Objects.requireNonNull(model, "model");
      switch (kind) {
        case PCS:
          handle = Utilities.INSTANCE.crispasr_pcs_init(model);
          break;
        case PUNCTUATION:
          handle = Utilities.INSTANCE.crispasr_punc_init(model);
          break;
        default:
          handle = Utilities.INSTANCE.crispasr_truecase_init(model);
      }
      if (handle == null)
        throw new IllegalStateException("text processor could not open");
    }
    public synchronized String process(String text) {
      if (handle == null)
        throw new IllegalStateException("text processor is closed");
      Objects.requireNonNull(text, "text");
      Pointer out;
      switch (kind) {
        case PCS:
          out = Utilities.INSTANCE.crispasr_pcs_process(handle, text);
          break;
        case PUNCTUATION:
          out = Utilities.INSTANCE.crispasr_punc_process(handle, text);
          break;
        default:
          out = Utilities.INSTANCE.crispasr_truecase_process(handle, text);
      }
      try {
        if (out == null)
          throw new IllegalStateException("text processing failed");
        return out.getString(0, "UTF-8");
      } finally {
        if (out != null) {
          switch (kind) {
            case PCS:
              Utilities.INSTANCE.crispasr_pcs_free_text(out);
              break;
            case PUNCTUATION:
              Utilities.INSTANCE.crispasr_punc_free_text(out);
              break;
            default:
              Utilities.INSTANCE.crispasr_truecase_free_text(out);
          }
        }
      }
    }
    public synchronized void close() {
      if (handle != null) {
        switch (kind) {
          case PCS:
            Utilities.INSTANCE.crispasr_pcs_free(handle);
            break;
          case PUNCTUATION:
            Utilities.INSTANCE.crispasr_punc_free(handle);
            break;
          default:
            Utilities.INSTANCE.crispasr_truecase_free(handle);
        }
        handle = null;
      }
    }
  }
  public static final class TextScores {
    public final float[] logProbabilities;
    public final int[] tokenCounts;
    TextScores(float[] scores, int[] counts) {
      logProbabilities = scores;
      tokenCounts = counts;
    }
  }
  public TextScores scoreTexts(float[] pcm, String[] candidates, String language, String prompt) {
    Objects.requireNonNull(pcm);
    Objects.requireNonNull(candidates);
    for (String t : candidates) Objects.requireNonNull(t);
    float[] scores = new float[candidates.length];
    int[] counts = new int[candidates.length];
    status(Utilities.INSTANCE.crispasr_session_score_texts(
        nativeHandle(), pcm, pcm.length, language, prompt, candidates, candidates.length, scores, counts));
    return new TextScores(scores, counts);
  }
}
