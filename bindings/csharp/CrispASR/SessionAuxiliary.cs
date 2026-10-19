using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace CrispASR
{
    internal static partial class NativeMethods
    {
        [StructLayout(LayoutKind.Sequential)] internal struct OpenParamsV2 { internal int Version, Threads, Gpu, Verbosity, Flash, GpuLayers; [MarshalAs(UnmanagedType.ByValArray, SizeConst = 6)] internal int[] Reserved; }
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_session_open_with_params([MarshalAs(UnmanagedType.LPUTF8Str)] string model, [MarshalAs(UnmanagedType.LPUTF8Str)] string? backend, ref OpenParamsV2 options);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_speaker_embedder_make_abi([MarshalAs(UnmanagedType.LPUTF8Str)] string spec, int threads, [MarshalAs(UnmanagedType.LPUTF8Str)] string? cache);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_speaker_embedder_free_abi(IntPtr handle);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_speaker_embedder_dim_abi(IntPtr handle);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_speaker_embedder_name_abi(IntPtr handle);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_speaker_embedder_embed_abi(IntPtr handle, float[] pcm, int n, [Out] float[] output);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_speaker_cluster_abi(float[] embeddings, int n, int dim, float threshold, int maxSpeakers, [Out] int[] labels);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_pyannote_cache_compute_abi(float[] pcm, int n, [MarshalAs(UnmanagedType.LPUTF8Str)] string model, int threads);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_pyannote_cache_free_abi(IntPtr cache);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_pyannote_cache_apply_abi(IntPtr cache, long offset, [In, Out] NativeDiarizationSegment[] segments, int n);
        [StructLayout(LayoutKind.Sequential)] internal struct NativeDiarizationSegment { internal long T0, T1; internal int Speaker, Padding; }
        [StructLayout(LayoutKind.Sequential)] internal struct NativeDiarizationOptions { internal int Method, Threads; internal long Offset; internal IntPtr Model, Embedder; internal int Min, Max, Count, Padding; }
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_diarize_segments_abi(float[] left, float[]? right, int n, int stereo, [In, Out] NativeDiarizationSegment[] segments, int count, ref NativeDiarizationOptions options);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_diarize_segments_turns_abi(float[] left, float[]? right, int n, int stereo, [In, Out] NativeDiarizationSegment[] segments, int count, ref NativeDiarizationOptions options, [Out] NativeDiarizationSegment[]? turns, int cap, out int nTurns);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_text_detect_language([MarshalAs(UnmanagedType.LPUTF8Str)] string text, [MarshalAs(UnmanagedType.LPUTF8Str)] string model, int threads, byte[] label, int cap, out float confidence);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_vad_slices([MarshalAs(UnmanagedType.LPUTF8Str)] string model, float[] pcm, int n, int rate, float threshold, int minSpeech, int minSilence, int pad, float maxChunk, int threads, out IntPtr spans);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_registry_lookup_by_filename_abi([MarshalAs(UnmanagedType.LPUTF8Str)] string name, byte[] filename, int filenameCap, byte[] url, int urlCap, byte[] size, int sizeCap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_registry_list_backends_abi(byte[] data, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)][return: MarshalAs(UnmanagedType.I1)] internal static extern bool crispasr_kokoro_lang_is_german_abi([MarshalAs(UnmanagedType.LPUTF8Str)] string language);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)][return: MarshalAs(UnmanagedType.I1)] internal static extern bool crispasr_kokoro_lang_has_native_voice_abi([MarshalAs(UnmanagedType.LPUTF8Str)] string language);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_lcs_dedup_prefix_count(int[] previous, int nPrevious, int[] current, int nCurrent, int minimum);
    }
    public sealed class SessionOptions
    {
        public int Threads { get; set; } = 4;
        public bool UseGpu { get; set; } = true;
        public int Verbosity { get; set; }
        public bool FlashAttention { get; set; } = true;
        public int GpuLayers { get; set; } = -1;
    }
    public sealed partial class Session
    {
        public static Session Open(string modelPath, SessionOptions options, string? backend = null)
        {
            ArgumentNullException.ThrowIfNull(modelPath); ArgumentNullException.ThrowIfNull(options);
            var native = new NativeMethods.OpenParamsV2 { Version = 2, Threads = options.Threads, Gpu = options.UseGpu ? 1 : 0, Verbosity = options.Verbosity, Flash = options.FlashAttention ? 1 : 0, GpuLayers = options.GpuLayers, Reserved = new int[6] };
            var p = NativeMethods.crispasr_session_open_with_params(modelPath, backend, ref native); if (p == IntPtr.Zero) throw new InvalidOperationException("Session open with options failed"); return new Session(p);
        }
        public static LanguageDetection DetectTextLanguage(string text, string model, int threads = 4)
        {
            ArgumentNullException.ThrowIfNull(text); ArgumentNullException.ThrowIfNull(model); var label = new byte[65536];
            StrictCheck(NativeMethods.crispasr_text_detect_language(text, model, threads, label, label.Length, out float confidence), "text_detect_language"); return new LanguageDetection(NativeMethods.NullTerminated(label), confidence);
        }
        /// <summary>Padded VAD slices in seconds. Model-load failure is an error, distinct from silence.</summary>
        public static VadSpan[] VadSlices(string model, float[] pcm, int sampleRate = 16000, float threshold = .5f, int minSpeechMs = 250, int minSilenceMs = 100, int speechPadMs = 0, float maxChunkSeconds = 0, int threads = 4)
        {
            ArgumentNullException.ThrowIfNull(model); ArgumentNullException.ThrowIfNull(pcm); if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
            int n = NativeMethods.crispasr_vad_slices(model, pcm, pcm.Length, sampleRate, threshold, minSpeechMs, minSilenceMs, speechPadMs, maxChunkSeconds, threads, out var p);
            try { if (n < 0) throw new InvalidOperationException($"vad_slices failed (rc={n})"); var data = Audio.CopyNative(p, checked(n * 2)); var spans = new VadSpan[n]; for (int i = 0; i < n; ++i) spans[i] = new VadSpan(data[i * 2], data[i * 2 + 1]); return spans; }
            finally { if (p != IntPtr.Zero) NativeMethods.crispasr_vad_free(p); }
        }
    }
    public sealed class SpeakerEmbedder : IDisposable
    {
        private IntPtr handle;
        private IntPtr Handle => handle == IntPtr.Zero ? throw new ObjectDisposedException(nameof(SpeakerEmbedder)) : handle;
        public SpeakerEmbedder(string model, int threads = 4, string? cacheDirectory = null) { ArgumentNullException.ThrowIfNull(model); if (model.Contains('\0') || (cacheDirectory?.Contains('\0') ?? false)) throw new ArgumentException("Speaker paths must not contain NUL"); handle = NativeMethods.crispasr_speaker_embedder_make_abi(model, threads, cacheDirectory); if (handle == IntPtr.Zero) throw new InvalidOperationException("Speaker embedder failed to open"); }
        public int Dimension => NativeMethods.crispasr_speaker_embedder_dim_abi(Handle);
        public string Name => NativeMethods.PtrToUtf8(NativeMethods.crispasr_speaker_embedder_name_abi(Handle)) ?? "";
        public float[] Embed(float[] pcm16k) { ArgumentNullException.ThrowIfNull(pcm16k); int dim = Dimension; if (dim <= 0) throw new InvalidOperationException("Invalid embedding dimension"); var output = new float[dim]; if (NativeMethods.crispasr_speaker_embedder_embed_abi(Handle, pcm16k, pcm16k.Length, output) != 1) throw new InvalidOperationException("Speaker embedding failed"); return output; }
        public static (int[] Labels, int ClusterCount) Cluster(float[] embeddings, int dimension, float mergeThreshold = .65f, int maxSpeakers = 8)
        {
            ArgumentNullException.ThrowIfNull(embeddings); if (dimension <= 0 || embeddings.Length == 0 || embeddings.Length % dimension != 0) throw new ArgumentException("Expected a nonempty frame-major embedding matrix");
            if (!float.IsFinite(mergeThreshold) || maxSpeakers <= 0) throw new ArgumentException("Invalid speaker clustering options");
            int n = embeddings.Length / dimension; var labels = new int[n];
            int count = NativeMethods.crispasr_speaker_cluster_abi(embeddings, n, dimension, mergeThreshold, maxSpeakers, labels);
            if (count < 0 || count > n) throw new InvalidOperationException("Invalid speaker clustering count");
            foreach (int label in labels) if (count == 0 ? label != -1 : label < 0 || label >= count) throw new InvalidOperationException("Invalid speaker clustering labels");
            return (labels, count);
        }
        public void Dispose() { if (handle != IntPtr.Zero) { NativeMethods.crispasr_speaker_embedder_free_abi(handle); handle = IntPtr.Zero; } }
    }
    public readonly struct DiarizationSegment
    {
        public double Start { get; }
        public double End { get; }
        public int Speaker { get; }
        public DiarizationSegment(double start, double end, int speaker = -1) { Start = start; End = end; Speaker = speaker; }
        internal static NativeMethods.NativeDiarizationSegment[] ToNative(IReadOnlyList<DiarizationSegment> segments)
        {
            ArgumentNullException.ThrowIfNull(segments); var result = new NativeMethods.NativeDiarizationSegment[segments.Count];
            for (int i = 0; i < result.Length; ++i) { var seg = segments[i]; if (!double.IsFinite(seg.Start) || !double.IsFinite(seg.End) || seg.Start < 0 || seg.End < seg.Start) throw new ArgumentException("Invalid segment timing"); result[i] = new NativeMethods.NativeDiarizationSegment { T0 = checked((long)Math.Round(seg.Start * 100)), T1 = checked((long)Math.Round(seg.End * 100)), Speaker = seg.Speaker }; }
            return result;
        }
        internal static DiarizationSegment[] FromNative(NativeMethods.NativeDiarizationSegment[] segments) { var result = new DiarizationSegment[segments.Length]; for (int i = 0; i < result.Length; ++i) result[i] = new DiarizationSegment(Session.Seconds(segments[i].T0), Session.Seconds(segments[i].T1), segments[i].Speaker); return result; }
    }
    public sealed class DiarizationOptions
    {
        public int Method { get; set; }
        public int Threads { get; set; } = 4;
        public double SliceOffsetSeconds { get; set; }
        public string? Model { get; set; }
        public string? Embedder { get; set; }
        public int MinSpeakers { get; set; }
        public int MaxSpeakers { get; set; }
        public int NumSpeakers { get; set; }
        internal NativeMethods.NativeDiarizationOptions Native()
        {
            if (Method < 0 || Method > 5 || !double.IsFinite(SliceOffsetSeconds)) throw new ArgumentException("Invalid diarization options");
            // Allocate both strings as one owned scope to cover a second allocation failure.
            var p = new NativeMethods.NativeDiarizationOptions { Method = Method, Threads = Threads, Offset = checked((long)Math.Round(SliceOffsetSeconds * 100)), Min = MinSpeakers, Max = MaxSpeakers, Count = NumSpeakers };
            try { p.Model = Model == null ? IntPtr.Zero : Marshal.StringToCoTaskMemUTF8(Model); p.Embedder = Embedder == null ? IntPtr.Zero : Marshal.StringToCoTaskMemUTF8(Embedder); return p; }
            catch { if (p.Model != IntPtr.Zero) Marshal.FreeCoTaskMem(p.Model); throw; }
        }
    }
    public static class Diarization
    {
        /// <summary>Assign speakers to caller segments; optional turns share the same absolute timeline. All public times are seconds.</summary>
        public static (DiarizationSegment[] Segments, DiarizationSegment[] Turns) Assign(float[] left16k, IReadOnlyList<DiarizationSegment> segments, DiarizationOptions options, float[]? right16k = null, bool includeTurns = true)
        {
            ArgumentNullException.ThrowIfNull(left16k); ArgumentNullException.ThrowIfNull(options); if (right16k != null && right16k.Length != left16k.Length) throw new ArgumentException("Stereo channel lengths differ");
            var input = DiarizationSegment.ToNative(segments); if (input.Length == 0) throw new ArgumentException("At least one segment required"); var p = options.Native();
            try
            {
                if (!includeTurns) { int rc = NativeMethods.crispasr_diarize_segments_abi(left16k, right16k, left16k.Length, right16k == null ? 0 : 1, input, input.Length, ref p); if (rc != 0) throw new InvalidOperationException($"diarize failed (rc={rc})"); return (DiarizationSegment.FromNative(input), Array.Empty<DiarizationSegment>()); }
                NativeMethods.NativeDiarizationSegment[]? turns = null;
                for (int attempt = 0; attempt < 4; ++attempt) { int rc = NativeMethods.crispasr_diarize_segments_turns_abi(left16k, right16k, left16k.Length, right16k == null ? 0 : 1, input, input.Length, ref p, turns, turns?.Length ?? 0, out int n); if (rc != 0 && rc != 2) throw new InvalidOperationException($"diarize turns failed (rc={rc})"); if (n < 0) throw new InvalidOperationException("Invalid turn count"); if (n == 0) { if (rc != 0) throw new InvalidOperationException($"diarize turns failed (rc={rc})"); return (DiarizationSegment.FromNative(input), Array.Empty<DiarizationSegment>()); } if (turns != null && n <= turns.Length && rc == 0) { Array.Resize(ref turns, n); return (DiarizationSegment.FromNative(input), DiarizationSegment.FromNative(turns)); } turns = new NativeMethods.NativeDiarizationSegment[n]; }
                throw new InvalidOperationException("Diarization turn count did not stabilize");
            }
            finally { if (p.Model != IntPtr.Zero) Marshal.FreeCoTaskMem(p.Model); if (p.Embedder != IntPtr.Zero) Marshal.FreeCoTaskMem(p.Embedder); }
        }
    }
    public sealed class PyannoteCache : IDisposable
    {
        private IntPtr handle;
        public PyannoteCache(float[] audio16k, string model, int threads = 4) { ArgumentNullException.ThrowIfNull(audio16k); ArgumentNullException.ThrowIfNull(model); if (audio16k.Length == 0 || model.Contains('\0')) throw new ArgumentException("Nonempty PCM and NUL-free model path required"); handle = NativeMethods.crispasr_pyannote_cache_compute_abi(audio16k, audio16k.Length, model, threads); if (handle == IntPtr.Zero) throw new InvalidOperationException("Pyannote cache compute failed"); }
        public DiarizationSegment[] Apply(IReadOnlyList<DiarizationSegment> segments, double sliceOffsetSeconds = 0) { if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(PyannoteCache)); if (!double.IsFinite(sliceOffsetSeconds)) throw new ArgumentException("Invalid offset"); var data = DiarizationSegment.ToNative(segments); int rc = NativeMethods.crispasr_pyannote_cache_apply_abi(handle, checked((long)Math.Round(sliceOffsetSeconds * 100)), data, data.Length); if (rc != 0) throw new InvalidOperationException($"Pyannote cache apply failed (rc={rc})"); return DiarizationSegment.FromNative(data); }
        public void Dispose() { if (handle != IntPtr.Zero) { NativeMethods.crispasr_pyannote_cache_free_abi(handle); handle = IntPtr.Zero; } }
    }
    public readonly record struct RegistryEntry(string Filename, string Url, string Size);
    public readonly record struct RegistryArtifact(int Kind, RegistryEntry Entry);
    public readonly record struct RegistryBundle(string Backend, string License, bool RequiresAcceptance, RegistryArtifact[] Artifacts);
    public static class ModelRegistry
    {
        public static RegistryEntry? Lookup(string name, bool byFilename = false)
        {
            ArgumentNullException.ThrowIfNull(name); var filename = new byte[65536]; var url = new byte[65536]; var size = new byte[65536];
            int rc = byFilename ? NativeMethods.crispasr_registry_lookup_by_filename_abi(name, filename, filename.Length, url, url.Length, size, size.Length) : NativeMethods.crispasr_registry_lookup_abi(name, filename, filename.Length, url, url.Length, size, size.Length);
            if (rc == 1) return null; if (rc != 0) throw new InvalidOperationException($"Registry lookup failed (rc={rc})"); return new RegistryEntry(NativeMethods.NullTerminated(filename), NativeMethods.NullTerminated(url), NativeMethods.NullTerminated(size));
        }
        public static string[] Backends() { var data = new byte[65536]; if (NativeMethods.crispasr_registry_list_backends_abi(data, data.Length) < 0) throw new InvalidOperationException("Registry list failed"); return NativeMethods.NullTerminated(data).Split(',', StringSplitOptions.RemoveEmptyEntries); }
        public static RegistryBundle? DefaultBundle(string backend)
        {
            ArgumentNullException.ThrowIfNull(backend); var name = new byte[65536]; var license = new byte[65536]; int n = NativeMethods.crispasr_registry_default_bundle_info_abi(backend, name, name.Length, license, license.Length, out int acceptance); if (n < 0) throw new InvalidOperationException("Registry bundle lookup failed"); if (n == 0 && name[0] == 0) return null;
            var artifacts = new RegistryArtifact[n]; for (int i = 0; i < n; ++i) { var filename = new byte[65536]; var url = new byte[65536]; var size = new byte[65536]; int rc = NativeMethods.crispasr_registry_default_bundle_artifact_abi(backend, i, out int kind, filename, filename.Length, url, url.Length, size, size.Length); if (rc != 0) throw new InvalidOperationException($"Registry artifact lookup failed (rc={rc})"); artifacts[i] = new RegistryArtifact(kind, new RegistryEntry(NativeMethods.NullTerminated(filename), NativeMethods.NullTerminated(url), NativeMethods.NullTerminated(size))); }
            return new RegistryBundle(NativeMethods.NullTerminated(name), NativeMethods.NullTerminated(license), acceptance != 0, artifacts);
        }
        public static string CacheDirectory(string? directory = null) { var data = new byte[65536]; if (NativeMethods.crispasr_cache_dir_abi(directory, data, data.Length) != 0) throw new InvalidOperationException("Cache directory lookup failed"); return NativeMethods.NullTerminated(data); }
        public static string EnsureFile(RegistryEntry entry, string? directory = null, bool quiet = true) { var data = new byte[65536]; if (NativeMethods.crispasr_cache_ensure_file_abi(entry.Filename, entry.Url, quiet ? 1 : 0, directory, data, data.Length) != 0) throw new InvalidOperationException("Cache download failed"); return NativeMethods.NullTerminated(data); }
    }
}

namespace CrispASR
{
    public static class TextUtilities
    {
        public static bool KokoroLanguageIsGerman(string language) { ArgumentNullException.ThrowIfNull(language); return NativeMethods.crispasr_kokoro_lang_is_german_abi(language); }
        public static bool KokoroLanguageHasNativeVoice(string language) { ArgumentNullException.ThrowIfNull(language); return NativeMethods.crispasr_kokoro_lang_has_native_voice_abi(language); }
        public static int LcsDedupPrefixCount(int[] previous, int[] current, int minLcsLength = 3) { ArgumentNullException.ThrowIfNull(previous); ArgumentNullException.ThrowIfNull(current); if (minLcsLength <= 0) throw new ArgumentOutOfRangeException(nameof(minLcsLength)); return NativeMethods.crispasr_lcs_dedup_prefix_count(previous, previous.Length, current, current.Length, minLcsLength); }
    }
}
