package io.github.ggerganov.whispercpp;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Objects;

/**
 * Auxiliary session APIs: registry, speaker embedder/clustering, pyannote cache, diarize
 * (segments+turns+options), text LID, VAD slices, and v2 open options. Backed by exact
 * C ABI signatures/POD layouts extracted from src/crispasr_c_api.cpp; diarize turn rc=2
 * is buffer-too-small and retries, positive 1/2 on 0-success functions are failures.
 */
public final class CrispasrAuxiliary {
  private CrispasrAuxiliary() {}

  private static String nullTerminated(byte[] buf) {
    int n = 0;
    while (n < buf.length && buf[n] != 0) n++;
    return new String(buf, 0, n, StandardCharsets.UTF_8);
  }

  private static Memory utf8(String s) {
    byte[] b = s.getBytes(StandardCharsets.UTF_8);
    Memory m = new Memory(b.length + 1);
    m.write(0, b, 0, b.length);
    m.setByte(b.length, (byte) 0);
    return m;
  }

  /** Closed-roster speaker database. Opening/enrollment require the native consent attestation. */
  public static final class SpeakerDb implements AutoCloseable {
    private Pointer handle;
    public SpeakerDb(String directory, String expectedNamesCsv, boolean consentAttested) {
      Objects.requireNonNull(directory, "directory");
      Objects.requireNonNull(expectedNamesCsv, "expectedNamesCsv");
      if (!consentAttested || expectedNamesCsv.trim().isEmpty())
        throw new IllegalArgumentException(
            "An explicit consent attestation and participant roster are required");
      handle = CrispasrSession.Lib.INSTANCE.crispasr_speaker_db_open(directory, expectedNamesCsv, 1);
      if (handle == null)
        throw new IllegalStateException("speaker_db_open failed");
    }
    private Pointer checkedHandle() {
      if (handle == null)
        throw new IllegalStateException("SpeakerDb is closed");
      return handle;
    }
    public synchronized int count() {
      return CrispasrSession.Lib.INSTANCE.crispasr_speaker_db_count(checkedHandle());
    }
    /** A negative native score means no match, even with a negative threshold. */
    public synchronized SpeakerMatch match(float[] embedding, float threshold) {
      Pointer db = checkedHandle();
      checkEmbedding(embedding);
      if (!Float.isFinite(threshold))
        throw new IllegalArgumentException("threshold must be finite");
      byte[] name = new byte[256];
      float score = CrispasrSession.Lib.INSTANCE.crispasr_speaker_db_match(
          db, embedding, embedding.length, threshold, name, name.length);
      return new SpeakerMatch(score < 0 || score < threshold ? null : nullTerminated(name), score);
    }
    public synchronized SpeakerMatch match(float[] embedding) {
      return match(embedding, 0.5f);
    }
    public static void enroll(String directory, String name, float[] embedding, boolean consentAttested) {
      Objects.requireNonNull(directory, "directory");
      Objects.requireNonNull(name, "name");
      checkEmbedding(embedding);
      if (!consentAttested)
        throw new IllegalArgumentException("An explicit consent attestation is required");
      int rc = CrispasrSession.Lib.INSTANCE.crispasr_speaker_db_enroll2(
          directory, name, embedding, embedding.length, 1);
      if (rc != 0)
        throw new IllegalStateException("speaker_db_enroll2 failed (rc=" + rc + ")");
    }
    private static void checkEmbedding(float[] embedding) {
      Objects.requireNonNull(embedding, "embedding");
      if (embedding.length == 0)
        throw new IllegalArgumentException("embedding must not be empty");
    }
    @Override
    public synchronized void close() {
      if (handle != null) {
        CrispasrSession.Lib.INSTANCE.crispasr_speaker_db_free(handle);
        handle = null;
      }
    }
  }
  public static final class SpeakerMatch {
    public final String name;
    public final float score;
    SpeakerMatch(String name, float score) {
      this.name = name;
      this.score = score;
    }
  }

  // -----------------------------------------------------------------
  // Registry
  // -----------------------------------------------------------------

  public static final class RegistryEntry {
    public final String filename;
    public final String url;
    public final String size;

    RegistryEntry(String filename, String url, String size) {
      this.filename = filename;
      this.url = url;
      this.size = size;
    }
  }

  public static final class RegistryArtifact {
    public final int kind;
    public final RegistryEntry entry;

    RegistryArtifact(int kind, RegistryEntry entry) {
      this.kind = kind;
      this.entry = entry;
    }
  }

  public static final class RegistryBundle {
    public final String backend;
    public final String license;
    public final boolean requiresAcceptance;
    public final RegistryArtifact[] artifacts;

    RegistryBundle(String backend, String license, boolean requiresAcceptance, RegistryArtifact[] artifacts) {
      this.backend = backend;
      this.license = license;
      this.requiresAcceptance = requiresAcceptance;
      this.artifacts = artifacts.clone();
    }
  }

  public static RegistryEntry registryLookup(String backend) {
    Objects.requireNonNull(backend, "backend");
    byte[] f = new byte[1024];
    byte[] u = new byte[1024];
    byte[] s = new byte[1024];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_registry_lookup_abi(
        backend, f, f.length, u, u.length, s, s.length);
    if (rc == 1)
      return null;
    if (rc != 0)
      throw new IllegalStateException("crispasr_registry_lookup_abi failed (rc=" + rc + ")");
    return new RegistryEntry(nullTerminated(f), nullTerminated(u), nullTerminated(s));
  }

  public static RegistryEntry registryLookupByFilename(String filename) {
    Objects.requireNonNull(filename, "filename");
    byte[] f = new byte[1024];
    byte[] u = new byte[1024];
    byte[] s = new byte[1024];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_registry_lookup_by_filename_abi(
        filename, f, f.length, u, u.length, s, s.length);
    if (rc == 1)
      return null;
    if (rc != 0)
      throw new IllegalStateException("crispasr_registry_lookup_by_filename_abi failed (rc=" + rc + ")");
    return new RegistryEntry(nullTerminated(f), nullTerminated(u), nullTerminated(s));
  }

  public static String[] registryListBackends() {
    byte[] out = new byte[65536];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_registry_list_backends_abi(out, out.length);
    if (rc < 0)
      throw new IllegalStateException("crispasr_registry_list_backends_abi failed (rc=" + rc + ")");
    String csv = nullTerminated(out);
    if (csv.isEmpty())
      return new String[0];
    return csv.split(",", -1);
  }

  public static RegistryBundle registryDefaultBundle(String backend) {
    Objects.requireNonNull(backend, "backend");
    byte[] outBackend = new byte[1024];
    byte[] outLicense = new byte[1024];
    int[] req = new int[1];
    int n = CrispasrSession.Lib.INSTANCE.crispasr_registry_default_bundle_info_abi(
        backend, outBackend, outBackend.length, outLicense, outLicense.length, req);
    if (n < 0)
      throw new IllegalStateException("crispasr_registry_default_bundle_info_abi failed (rc=" + n + ")");
    if (n == 0 && outBackend[0] == 0)
      return null;
    RegistryArtifact[] arts = new RegistryArtifact[n];
    for (int i = 0; i < n; i++) {
      byte[] f = new byte[1024];
      byte[] u = new byte[1024];
      byte[] s = new byte[1024];
      int[] kind = new int[1];
      int rc = CrispasrSession.Lib.INSTANCE.crispasr_registry_default_bundle_artifact_abi(
          backend, i, kind, f, f.length, u, u.length, s, s.length);
      if (rc != 0)
        throw new IllegalStateException(
            "crispasr_registry_default_bundle_artifact_abi failed (rc=" + rc + ")");
      arts[i] = new RegistryArtifact(
          kind[0], new RegistryEntry(nullTerminated(f), nullTerminated(u), nullTerminated(s)));
    }
    return new RegistryBundle(nullTerminated(outBackend), nullTerminated(outLicense), req[0] != 0, arts);
  }

  public static String cacheDir(String override) {
    byte[] out = new byte[4096];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_cache_dir_abi(override, out, out.length);
    if (rc != 0)
      throw new IllegalStateException("crispasr_cache_dir_abi failed (rc=" + rc + ")");
    return nullTerminated(out);
  }

  public static String cacheEnsureFile(String filename, String url, boolean quiet, String cacheDirOverride) {
    Objects.requireNonNull(filename, "filename");
    Objects.requireNonNull(url, "url");
    byte[] out = new byte[4096];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_cache_ensure_file_abi(
        filename, url, quiet ? 1 : 0, cacheDirOverride, out, out.length);
    if (rc != 0)
      throw new IllegalStateException("crispasr_cache_ensure_file_abi failed (rc=" + rc + ")");
    return nullTerminated(out);
  }

  public static String cacheEnsureFile(RegistryEntry entry, String cacheDirOverride, boolean quiet) {
    Objects.requireNonNull(entry, "entry");
    return cacheEnsureFile(entry.filename, entry.url, quiet, cacheDirOverride);
  }

  // -----------------------------------------------------------------
  // V2 open options
  // -----------------------------------------------------------------

  public static final class SessionOptions {
    public int threads = 4;
    public boolean useGpu = true;
    public int verbosity = 0;
    public boolean flashAttention = true;
    public int gpuLayers = -1;

    public SessionOptions threads(int v) {
      if (v <= 0)
        throw new IllegalArgumentException("threads must be positive");
      this.threads = v;
      return this;
    }

    public SessionOptions useGpu(boolean v) {
      this.useGpu = v;
      return this;
    }

    public SessionOptions verbosity(int v) {
      this.verbosity = v;
      return this;
    }

    public SessionOptions flashAttention(boolean v) {
      this.flashAttention = v;
      return this;
    }

    public SessionOptions gpuLayers(int v) {
      this.gpuLayers = v;
      return this;
    }
  }

  public static CrispasrSession openWithOptions(String modelPath, String backend, SessionOptions opts) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(opts, "options");
    // Layout matches crispasr_open_params_v1 in src/crispasr_c_api.cpp: abi_version,
    // n_threads, use_gpu, verbosity, flash_attn, n_gpu_layers, reserved[6] => 12 ints = 48 bytes.
    Memory p = new Memory(48);
    p.clear();
    p.setInt(0, 2); // abi_version
    p.setInt(4, opts.threads);
    p.setInt(8, opts.useGpu ? 1 : 0);
    p.setInt(12, opts.verbosity);
    p.setInt(16, opts.flashAttention ? 1 : 0);
    p.setInt(20, opts.gpuLayers);
    // reserved[6] already zero
    Pointer h = CrispasrSession.Lib.INSTANCE.crispasr_session_open_with_params(modelPath, backend, p);
    if (h == null)
      throw new IllegalStateException("crispasr_session_open_with_params failed for " + modelPath);
    return CrispasrSession.wrap(h);
  }

  // -----------------------------------------------------------------
  // Speaker embedder
  // -----------------------------------------------------------------

  public static final class SpeakerEmbedder implements AutoCloseable {
    private Pointer handle;

    public SpeakerEmbedder(String modelSpec, int nThreads, String cacheDir) {
      Objects.requireNonNull(modelSpec, "modelSpec");
      if (modelSpec.indexOf(0) >= 0 || (cacheDir != null && cacheDir.indexOf(0) >= 0))
        throw new IllegalArgumentException("Speaker paths must not contain NUL");
      if (modelSpec.isEmpty())
        throw new IllegalArgumentException("modelSpec empty");
      if (nThreads <= 0)
        throw new IllegalArgumentException("nThreads must be positive");
      Pointer p =
          CrispasrSession.Lib.INSTANCE.crispasr_speaker_embedder_make_abi(modelSpec, nThreads, cacheDir);
      if (p == null)
        throw new IllegalStateException("crispasr_speaker_embedder_make_abi failed");
      this.handle = p;
    }

    public SpeakerEmbedder(String modelSpec) {
      this(modelSpec, 4, null);
    }

    private void checkOpen() {
      if (handle == null)
        throw new IllegalStateException("SpeakerEmbedder is closed");
    }

    public int dim() {
      checkOpen();
      int d = CrispasrSession.Lib.INSTANCE.crispasr_speaker_embedder_dim_abi(handle);
      if (d <= 0)
        throw new IllegalStateException("invalid embedder dim: " + d);
      return d;
    }

    public String name() {
      checkOpen();
      String s = CrispasrSession.Lib.INSTANCE.crispasr_speaker_embedder_name_abi(handle);
      return s == null ? "" : s;
    }

    public float[] embed(float[] pcm16k) {
      checkOpen();
      Objects.requireNonNull(pcm16k, "pcm16k");
      if (pcm16k.length == 0)
        throw new IllegalArgumentException("pcm empty");
      int d = dim();
      float[] out = new float[d];
      int rc = CrispasrSession.Lib.INSTANCE.crispasr_speaker_embedder_embed_abi(
          handle, pcm16k, pcm16k.length, out);
      if (rc != 1)
        throw new IllegalStateException("crispasr_speaker_embedder_embed_abi failed (rc=" + rc + ")");
      return out.clone();
    }

    @Override
    public void close() {
      if (handle != null) {
        CrispasrSession.Lib.INSTANCE.crispasr_speaker_embedder_free_abi(handle);
        handle = null;
      }
    }
  }

  public static final class ClusterResult {
    public final int[] labels;
    public final int clusterCount;

    ClusterResult(int[] labels, int clusterCount) {
      this.labels = labels.clone();
      this.clusterCount = clusterCount;
    }
  }

  public static ClusterResult cluster(float[] embeddings, int dim, float mergeThreshold, int maxSpeakers) {
    Objects.requireNonNull(embeddings, "embeddings");
    if (dim <= 0)
      throw new IllegalArgumentException("dim must be positive");
    if (embeddings.length == 0 || embeddings.length % dim != 0)
      throw new IllegalArgumentException("embeddings length must be multiple of dim");
    if (!Float.isFinite(mergeThreshold))
      throw new IllegalArgumentException("mergeThreshold must be finite");
    if (maxSpeakers <= 0)
      throw new IllegalArgumentException("maxSpeakers must be positive");
    int n = embeddings.length / dim;
    int[] labels = new int[n];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_speaker_cluster_abi(
        embeddings, n, dim, mergeThreshold, maxSpeakers, labels);
    if (rc < 0 || rc > n)
      throw new IllegalStateException("Invalid speaker cluster count: " + rc);
    for (int label : labels) {
      if (rc == 0 ? label != -1 : label < 0 || label >= rc)
        throw new IllegalStateException("Invalid speaker cluster label: " + label);
    }
    return new ClusterResult(labels, rc);
  }

  // -----------------------------------------------------------------
  // Pyannote cache
  // -----------------------------------------------------------------

  public static final class PyannoteCache implements AutoCloseable {
    private Pointer handle;

    public PyannoteCache(float[] fullAudio, String modelPath, int nThreads) {
      Objects.requireNonNull(fullAudio, "fullAudio");
      Objects.requireNonNull(modelPath, "modelPath");
      if (modelPath.indexOf(0) >= 0)
        throw new IllegalArgumentException("Model path must not contain NUL");
      if (fullAudio.length == 0)
        throw new IllegalArgumentException("fullAudio empty");
      if (nThreads <= 0)
        throw new IllegalArgumentException("nThreads must be positive");
      Pointer p = CrispasrSession.Lib.INSTANCE.crispasr_pyannote_cache_compute_abi(
          fullAudio, fullAudio.length, modelPath, nThreads);
      if (p == null)
        throw new IllegalStateException("crispasr_pyannote_cache_compute_abi failed");
      this.handle = p;
    }

    public PyannoteCache(float[] fullAudio, String modelPath) {
      this(fullAudio, modelPath, 4);
    }

    private void checkOpen() {
      if (handle == null)
        throw new IllegalStateException("PyannoteCache is closed");
    }

    public DiarizationSegment[] apply(DiarizationSegment[] segments, long sliceT0Cs) {
      checkOpen();
      Objects.requireNonNull(segments, "segments");
      if (segments.length == 0)
        throw new IllegalArgumentException("segments empty");
      if (!isFiniteLong(sliceT0Cs))
        throw new IllegalArgumentException("sliceT0Cs not finite");
      for (DiarizationSegment s : segments) validateSegment(s);
      Memory segsMem = packSegments(segments);
      try {
        int rc = CrispasrSession.Lib.INSTANCE.crispasr_pyannote_cache_apply_abi(
            handle, sliceT0Cs, segsMem, segments.length);
        if (rc != 0)
          throw new IllegalStateException("crispasr_pyannote_cache_apply_abi failed (rc=" + rc + ")");
        return unpackSegments(segsMem, segments.length);
      } finally {
        // segsMem will be GC'd; no native free needed (stack buffer for pyannote)
      }
    }

    @Override
    public void close() {
      if (handle != null) {
        CrispasrSession.Lib.INSTANCE.crispasr_pyannote_cache_free_abi(handle);
        handle = null;
      }
    }
  }

  // -----------------------------------------------------------------
  // Diarize: POD layouts extracted literal from src/crispasr_c_api.cpp
  //   crispasr_diarize_seg_abi  : int64 t0_cs, int64 t1_cs, int32 speaker, int32 _pad = 24 bytes
  //   crispasr_diarize_turn_abi : same 24 bytes
  //   crispasr_diarize_opts_abi : int32 method, int32 n_threads, int64 slice_t0_cs,
  //                               const char* pyannote_model_path, const char* foxnose_embedder_path,
  //                               int32 min_speakers, int32 max_speakers, int32 num_speakers, int32 _pad2 =
  //                               48 bytes
  // -----------------------------------------------------------------

  public static final class DiarizationSegment {
    public final long t0; // centiseconds
    public final long t1; // centiseconds
    public final int speaker; // -1 unassigned

    public DiarizationSegment(long t0, long t1, int speaker) {
      this.t0 = t0;
      this.t1 = t1;
      this.speaker = speaker;
    }

    public DiarizationSegment(long t0, long t1) {
      this(t0, t1, -1);
    }
  }

  public static final class DiarizationOptions {
    public int method = 0;
    public int nThreads = 4;
    public long sliceOffsetCs = 0;
    public String pyannoteModelPath = null;
    public String foxNoseEmbedderPath = null;
    public int minSpeakers = 0;
    public int maxSpeakers = 0;
    public int numSpeakers = 0;

    public DiarizationOptions method(int v) {
      this.method = v;
      return this;
    }

    public DiarizationOptions nThreads(int v) {
      this.nThreads = v;
      return this;
    }

    public DiarizationOptions sliceOffsetCs(long v) {
      this.sliceOffsetCs = v;
      return this;
    }

    public DiarizationOptions sliceOffsetSeconds(double seconds) {
      if (!Double.isFinite(seconds))
        throw new IllegalArgumentException("slice offset not finite");
      this.sliceOffsetCs = Math.round(seconds * 100.0);
      return this;
    }

    public DiarizationOptions pyannoteModelPath(String v) {
      this.pyannoteModelPath = v;
      return this;
    }

    public DiarizationOptions foxNoseEmbedderPath(String v) {
      this.foxNoseEmbedderPath = v;
      return this;
    }

    public DiarizationOptions minSpeakers(int v) {
      this.minSpeakers = v;
      return this;
    }

    public DiarizationOptions maxSpeakers(int v) {
      this.maxSpeakers = v;
      return this;
    }

    public DiarizationOptions numSpeakers(int v) {
      this.numSpeakers = v;
      return this;
    }
  }

  public static final class DiarizationResult {
    public final DiarizationSegment[] segments;
    public final DiarizationSegment[] turns;

    DiarizationResult(DiarizationSegment[] segments, DiarizationSegment[] turns) {
      this.segments = segments.clone();
      this.turns = turns.clone();
    }
  }

  private static boolean isFiniteLong(long v) {
    return true;
  }

  private static void validateSegment(DiarizationSegment s) {
    Objects.requireNonNull(s, "segment");
    if (s.t0 < 0 || s.t1 < s.t0)
      throw new IllegalArgumentException("invalid segment timing: t0=" + s.t0 + " t1=" + s.t1);
  }

  private static Memory packSegments(DiarizationSegment[] segs) {
    Memory mem = new Memory((long) segs.length * 24L);
    mem.clear();
    for (int i = 0; i < segs.length; i++) {
      long off = (long) i * 24L;
      mem.setLong(off, segs[i].t0);
      mem.setLong(off + 8, segs[i].t1);
      mem.setInt(off + 16, segs[i].speaker);
      mem.setInt(off + 20, 0);
    }
    return mem;
  }

  private static DiarizationSegment[] unpackSegments(Pointer mem, int n) {
    DiarizationSegment[] out = new DiarizationSegment[n];
    for (int i = 0; i < n; i++) {
      long off = (long) i * 24L;
      long t0 = mem.getLong(off);
      long t1 = mem.getLong(off + 8);
      int spk = mem.getInt(off + 16);
      out[i] = new DiarizationSegment(t0, t1, spk);
    }
    return out;
  }

  private static Memory packOpts(DiarizationOptions opts, Memory[] toKeep) {
    if (opts.method < 0 || opts.method > 5)
      throw new IllegalArgumentException("method out of range 0..5: " + opts.method);
    if (!Double.isFinite((double) opts.sliceOffsetCs))
      throw new IllegalArgumentException("slice offset not finite");
    int embedderOffset = 16 + Native.POINTER_SIZE;
    int speakersOffset = 16 + 2 * Native.POINTER_SIZE;
    Memory mem = new Memory(speakersOffset + 16);
    mem.clear();
    mem.setInt(0, opts.method);
    mem.setInt(4, opts.nThreads);
    mem.setLong(8, opts.sliceOffsetCs);
    Memory modelMem = null;
    Memory embedMem = null;
    if (opts.pyannoteModelPath != null) {
      modelMem = utf8(opts.pyannoteModelPath);
      toKeep[0] = modelMem;
      mem.setPointer(16, modelMem);
    } else {
      mem.setPointer(16, null);
    }
    if (opts.foxNoseEmbedderPath != null) {
      embedMem = utf8(opts.foxNoseEmbedderPath);
      toKeep[1] = embedMem;
      mem.setPointer(embedderOffset, embedMem);
    } else {
      mem.setPointer(embedderOffset, null);
    }
    mem.setInt(speakersOffset, opts.minSpeakers);
    mem.setInt(speakersOffset + 4, opts.maxSpeakers);
    mem.setInt(speakersOffset + 8, opts.numSpeakers);
    mem.setInt(speakersOffset + 12, 0);
    return mem;
  }

  public static DiarizationResult diarize(
      float[] leftPcm, float[] rightPcm, DiarizationSegment[] segments, DiarizationOptions opts) {
    return diarizeInternal(leftPcm, rightPcm, segments, opts, true);
  }

  public static DiarizationResult diarizeWithoutTurns(
      float[] leftPcm, DiarizationSegment[] segments, DiarizationOptions opts) {
    return diarizeInternal(leftPcm, null, segments, opts, false);
  }

  private static DiarizationResult diarizeInternal(float[] leftPcm, float[] rightPcm,
      DiarizationSegment[] segments, DiarizationOptions opts, boolean includeTurns) {
    Objects.requireNonNull(leftPcm, "leftPcm");
    Objects.requireNonNull(segments, "segments");
    Objects.requireNonNull(opts, "opts");
    if (leftPcm.length == 0)
      throw new IllegalArgumentException("leftPcm empty");
    if (segments.length == 0)
      throw new IllegalArgumentException("segments empty");
    for (DiarizationSegment s : segments) validateSegment(s);
    if (rightPcm != null && rightPcm.length != leftPcm.length)
      throw new IllegalArgumentException("stereo channel lengths differ");
    int isStereo = (rightPcm != null) ? 1 : 0;

    Memory segsMem = packSegments(segments);
    Memory[] keep = new Memory[2];
    Memory optsMem = packOpts(opts, keep);
    try {
      if (!includeTurns) {
        int rc = CrispasrSession.Lib.INSTANCE.crispasr_diarize_segments_abi(
            leftPcm, rightPcm, leftPcm.length, isStereo, segsMem, segments.length, optsMem);
        if (rc != 0)
          throw new IllegalStateException("crispasr_diarize_segments_abi failed (rc=" + rc + ")");
        DiarizationSegment[] outSegs = unpackSegments(segsMem, segments.length);
        return new DiarizationResult(outSegs, new DiarizationSegment[0]);
      } else {
        // Retry loop for rc=2 buffer-too-small (turns). Never falsely succeed.
        Memory turnsMem = null;
        IntByReference outNTurns = new IntByReference();
        // First try with null buffer to get count? C# does 4 attempts starting with null.
        // We replicate: attempt with current capacity, on rc=2 or need > cap, enlarge.
        int cap = 0;
        Memory curTurns = null;
        for (int attempt = 0; attempt < 4; attempt++) {
          int rc = CrispasrSession.Lib.INSTANCE.crispasr_diarize_segments_turns_abi(leftPcm, rightPcm,
              leftPcm.length, isStereo, segsMem, segments.length, optsMem, curTurns, cap, outNTurns);
          if (rc != 0 && rc != 2)
            throw new IllegalStateException("crispasr_diarize_segments_turns_abi failed (rc=" + rc + ")");
          int need = outNTurns.getValue();
          if (need < 0)
            throw new IllegalStateException("invalid turn count: " + need);
          if (need == 0) {
            if (rc != 0)
              throw new IllegalStateException("Invalid empty turn result (rc=" + rc + ")");
            DiarizationSegment[] outSegs = unpackSegments(segsMem, segments.length);
            return new DiarizationResult(outSegs, new DiarizationSegment[0]);
          }
          if (curTurns != null && need <= cap && rc == 0) {
            DiarizationSegment[] outSegs = unpackSegments(segsMem, segments.length);
            DiarizationSegment[] turns = new DiarizationSegment[need];
            for (int i = 0; i < need; i++) {
              long off = (long) i * 24L;
              long t0 = curTurns.getLong(off);
              long t1 = curTurns.getLong(off + 8);
              int spk = curTurns.getInt(off + 16);
              turns[i] = new DiarizationSegment(t0, t1, spk);
            }
            return new DiarizationResult(outSegs, turns);
          }
          if (attempt == 3)
            throw new IllegalStateException("diarization turn count did not stabilize");
          cap = need;
          curTurns = new Memory((long) cap * 24L);
          curTurns.clear();
          // On next iteration we need to reuse segsMem that already has labeled speakers,
          // but the native impl re-labels each call, so we must repack original segments
          // to avoid carrying over mutated speaker labels into next attempt's input check.

          segsMem = packSegments(segments);
        }
        throw new IllegalStateException("unreachable diarize retry");
      }
    } finally {
      // keep Memory for model/embed freed by GC
    }
  }

  // -----------------------------------------------------------------
  // Text language detection
  // -----------------------------------------------------------------

  public static final class LanguageDetection {
    public final String label;
    public final float confidence;

    LanguageDetection(String label, float confidence) {
      this.label = label;
      this.confidence = confidence;
    }
  }

  public static LanguageDetection detectTextLanguage(String text, String modelPath, int nThreads) {
    Objects.requireNonNull(text, "text");
    Objects.requireNonNull(modelPath, "modelPath");
    if (nThreads <= 0)
      throw new IllegalArgumentException("nThreads must be positive");
    byte[] outLabel = new byte[4096];
    float[] outConf = new float[1];
    int rc = CrispasrSession.Lib.INSTANCE.crispasr_text_detect_language(
        text, modelPath, nThreads, outLabel, outLabel.length, outConf);
    if (rc != 0)
      throw new IllegalStateException("crispasr_text_detect_language failed (rc=" + rc + ")");
    return new LanguageDetection(nullTerminated(outLabel), outConf[0]);
  }

  public static LanguageDetection detectTextLanguage(String text, String modelPath) {
    return detectTextLanguage(text, modelPath, 4);
  }

  // -----------------------------------------------------------------
  // VAD slices (seconds)
  // -----------------------------------------------------------------

  public static CrispasrSession.VADSpan[] vadSlices(String modelPath, float[] pcm, int sampleRate,
      float threshold, int minSpeechMs, int minSilenceMs, int speechPadMs, float maxChunkSec, int nThreads) {
    Objects.requireNonNull(modelPath, "modelPath");
    Objects.requireNonNull(pcm, "pcm");
    if (sampleRate <= 0)
      throw new IllegalArgumentException("sampleRate must be positive");
    if (nThreads <= 0)
      throw new IllegalArgumentException("nThreads must be positive");
    if (pcm.length == 0)
      return new CrispasrSession.VADSpan[0];
    Pointer[] outSpans = new Pointer[1];
    int n = CrispasrSession.Lib.INSTANCE.crispasr_vad_slices(modelPath, pcm, pcm.length, sampleRate,
        threshold, minSpeechMs, minSilenceMs, speechPadMs, maxChunkSec, nThreads, outSpans);
    try {
      if (n < 0)
        throw new IllegalStateException("crispasr_vad_slices failed (rc=" + n + ")");
      if (n == 0 || outSpans[0] == null)
        return new CrispasrSession.VADSpan[0];
      float[] raw = outSpans[0].getFloatArray(0, n * 2);
      // Copy before freeing
      float[] copy = Arrays.copyOf(raw, raw.length);
      CrispasrSession.VADSpan[] spans = new CrispasrSession.VADSpan[n];
      for (int i = 0; i < n; i++) spans[i] = new CrispasrSession.VADSpan(copy[i * 2], copy[i * 2 + 1]);
      return spans;
    } finally {
      if (outSpans[0] != null)
        CrispasrSession.Lib.INSTANCE.crispasr_vad_free(outSpans[0]);
    }
  }

  public static CrispasrSession.VADSpan[] vadSlices(String modelPath, float[] pcm) {
    return vadSlices(modelPath, pcm, 16000, 0.5f, 250, 100, 30, 0f, 4);
  }

  // -----------------------------------------------------------------
  // LID audio helper (wraps public header if needed) - already via Session?
  // -----------------------------------------------------------------

  // -----------------------------------------------------------------
  // LCS dedup helper - thin wrapper, validates
  // -----------------------------------------------------------------

  public static int lcsDedupPrefixCount(int[] prev, int[] curr, int minLcs) {
    Objects.requireNonNull(prev, "prev");
    Objects.requireNonNull(curr, "curr");
    if (minLcs <= 0)
      throw new IllegalArgumentException("minLcs must be positive");
    return CrispasrSession.Lib.INSTANCE.crispasr_lcs_dedup_prefix_count(
        prev, prev.length, curr, curr.length, minLcs);
  }

  /** Legacy Parakeet decoder. Times/offsets are centiseconds; all output is copied. */
  public static final class Parakeet implements AutoCloseable {
    private Pointer handle;
    public Parakeet(String model, int threads, boolean flash) {
      handle = CrispasrSession.Lib.INSTANCE.crispasr_parakeet_init(
          Objects.requireNonNull(model), threads, flash ? 1 : 0);
      if (handle == null)
        throw new IllegalStateException("Parakeet model could not open");
    }
    public synchronized ParakeetTranscript transcribe(float[] pcm, long offsetCs) {
      Objects.requireNonNull(pcm);
      if (handle == null)
        throw new IllegalStateException("Parakeet context closed");
      if (pcm.length == 0)
        throw new IllegalArgumentException("Nonempty PCM required");
      CrispasrSession.Lib lib = CrispasrSession.Lib.INSTANCE;
      Pointer r = lib.crispasr_parakeet_transcribe(handle, pcm, pcm.length, offsetCs);
      if (r == null)
        throw new IllegalStateException("Parakeet transcription failed");
      try {
        int nw = lib.crispasr_parakeet_result_n_words(r), nt = lib.crispasr_parakeet_result_n_tokens(r);
        if (nw < 0 || nt < 0)
          throw new IllegalStateException("Invalid Parakeet count");
        ParakeetWord[] words = new ParakeetWord[nw];
        CrispasrSession.SessionToken[] tokens = new CrispasrSession.SessionToken[nt];
        for (int j = 0; j < nw; ++j)
          words[j] = new ParakeetWord(lib.crispasr_parakeet_result_word_text(r, j),
              lib.crispasr_parakeet_result_word_t0(r, j), lib.crispasr_parakeet_result_word_t1(r, j));
        for (int j = 0; j < nt; ++j)
          tokens[j] = new CrispasrSession.SessionToken(
              CrispasrSession.copyUtf8Bytes(lib.crispasr_parakeet_result_token_text(r, j)),
              lib.crispasr_parakeet_result_token_t0(r, j), lib.crispasr_parakeet_result_token_t1(r, j),
              lib.crispasr_parakeet_result_token_p(r, j));
        return new ParakeetTranscript(lib.crispasr_parakeet_result_text(r), words, tokens);
      } finally {
        lib.crispasr_parakeet_result_free(r);
      }
    }
    public synchronized void close() {
      if (handle != null) {
        CrispasrSession.Lib.INSTANCE.crispasr_parakeet_free(handle);
        handle = null;
      }
    }
  }
  public static final class ParakeetWord {
    public final String text;
    public final long t0, t1;
    ParakeetWord(String text, long t0, long t1) {
      this.text = text;
      this.t0 = t0;
      this.t1 = t1;
    }
  }
  public static final class ParakeetTranscript {
    public final String text;
    public final ParakeetWord[] words;
    public final CrispasrSession.SessionToken[] tokens;
    ParakeetTranscript(String text, ParakeetWord[] words, CrispasrSession.SessionToken[] tokens) {
      this.text = text;
      this.words = words;
      this.tokens = tokens;
    }
  }

  /** Direct TitaNet model. Native embed returns a float count, exactly 192. */
  public static final class TitaNet implements AutoCloseable {
    private Pointer handle;
    public TitaNet(String model, int threads) {
      handle = CrispasrSession.Lib.INSTANCE.crispasr_titanet_init(Objects.requireNonNull(model), threads);
      if (handle == null)
        throw new IllegalStateException("TitaNet model could not open");
    }
    public synchronized float[] embed(float[] pcm) {
      Objects.requireNonNull(pcm);
      if (handle == null)
        throw new IllegalStateException("TitaNet context closed");
      if (pcm.length == 0)
        throw new IllegalArgumentException("Nonempty PCM required");
      float[] output = new float[192];
      int n = CrispasrSession.Lib.INSTANCE.crispasr_titanet_embed(handle, pcm, pcm.length, output);
      if (n != 192)
        throw new IllegalStateException("TitaNet must return 192 floats (got " + n + ")");
      return output;
    }
    public static float cosineSimilarity(float[] a, float[] b) {
      Objects.requireNonNull(a);
      Objects.requireNonNull(b);
      if (a.length != b.length)
        throw new IllegalArgumentException("Embedding dimensions must match");
      return CrispasrSession.Lib.INSTANCE.crispasr_titanet_cosine_sim(a, b, a.length);
    }
    public synchronized void close() {
      if (handle != null) {
        CrispasrSession.Lib.INSTANCE.crispasr_titanet_free(handle);
        handle = null;
      }
    }
  }
}
