using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace CrispASR
{
    internal static partial class NativeMethods
    {
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_translate(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_detect_language(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_token_timestamps(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_split_on_word(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_no_context(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_single_segment(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_print_realtime(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_print_progress(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_print_timestamps(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_print_special(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_suppress_blank(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_vad(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_tdrz(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_n_threads(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_max_len(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_best_of(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_max_tokens(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_alt_n(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_vad_min_speech_ms(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_vad_min_silence_ms(IntPtr p, int value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_temperature(IntPtr p, float value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_vad_threshold(IntPtr p, float value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_language(IntPtr p, IntPtr value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_initial_prompt(IntPtr p, IntPtr value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_params_set_vad_model_path(IntPtr p, IntPtr value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr whisper_context_default_params_by_ref();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void whisper_free_context_params(IntPtr p);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr whisper_full_default_params_by_ref(int strategy);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void whisper_free_params(IntPtr p);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr whisper_init_from_file_with_params_by_ref([MarshalAs(UnmanagedType.LPUTF8Str)] string model, IntPtr p);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void whisper_free(IntPtr ctx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int whisper_full_by_ref(IntPtr ctx, IntPtr p, float[] pcm, int count);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_transcribe_parallel_by_ref(IntPtr ctx, IntPtr p, float[] pcm, int count, int processors);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_ctx_params_set_dtw(IntPtr p, [MarshalAs(UnmanagedType.I1)] bool enabled, int preset, int top);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int whisper_full_n_segments(IntPtr ctx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr whisper_full_get_segment_text(IntPtr ctx, int segment);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long whisper_full_get_segment_t0(IntPtr ctx, int segment);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long whisper_full_get_segment_t1(IntPtr ctx, int segment);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern float whisper_full_get_segment_no_speech_prob(IntPtr ctx, int segment);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int whisper_full_n_tokens(IntPtr ctx, int segment);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr whisper_full_get_token_text(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int whisper_full_get_token_id(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_token_t0(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_token_t1(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern float crispasr_token_p(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_token_dtw_t(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_token_n_alts(IntPtr ctx, int segment, int token);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_token_alt_id(IntPtr ctx, int segment, int token, int alternative);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern float crispasr_token_alt_p(IntPtr ctx, int segment, int token, int alternative);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_token_alt_text(IntPtr ctx, int segment, int token, int alternative, [Out] byte[] text, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern float crispasr_detect_language(IntPtr ctx, float[] pcm, int count, int threads, byte[] language, int cap);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_stream_open(IntPtr ctx, int threads, int step, int length, int keep, [MarshalAs(UnmanagedType.LPUTF8Str)] string language, int translate);
    }
    public sealed class WhisperDecodeOptions
    {
        public int Strategy { get; set; }
        public int Processors { get; set; } = 1;
        public bool Silent { get; set; } = true;
        public string? Language { get; set; } = "en";
        public string? InitialPrompt { get; set; }
        public string? VadModelPath { get; set; }
        public bool? Translate { get; set; }
        public bool? DetectLanguage { get; set; }
        public bool? TokenTimestamps { get; set; }
        public bool? SplitOnWord { get; set; }
        public bool? NoContext { get; set; }
        public bool? SingleSegment { get; set; }
        public bool? PrintRealtime { get; set; }
        public bool? PrintProgress { get; set; }
        public bool? PrintTimestamps { get; set; }
        public bool? PrintSpecial { get; set; }
        public bool? SuppressBlank { get; set; }
        public bool? Vad { get; set; }
        public bool? Tdrz { get; set; }
        public int? NThreads { get; set; }
        public int? MaxLen { get; set; }
        public int? BestOf { get; set; }
        public int? MaxTokens { get; set; }
        public int? AltN { get; set; }
        public int? VadMinSpeechMs { get; set; }
        public int? VadMinSilenceMs { get; set; }
        public float? Temperature { get; set; }
        public float? VadThreshold { get; set; }
    }
    public readonly record struct WhisperAlternative(int Id, string Text, float Probability, byte[] RawText);
    public readonly record struct WhisperToken(int Id, string Text, byte[] RawText, double Start, double End, float Probability, double DtwTime, WhisperAlternative[] Alternatives)
    {
        public bool HasTimestamps => Start >= 0 && End >= Start;
        public bool HasDtwTimestamp => DtwTime >= 0;
    }
    public readonly record struct WhisperSegment(string Text, double Start, double End, float NoSpeechProbability, WhisperToken[] Tokens);
    /// <summary>Legacy Whisper controls using opaque native parameters and pointer adapters. One operation at a time per context.</summary>
    public sealed class WhisperContext : IDisposable
    {
        private IntPtr handle;
        private readonly object gate = new();
        private readonly HashSet<StreamDecoder> streams = new();
        private WhisperContext(IntPtr native) { handle = native; }
        private IntPtr Handle => handle == IntPtr.Zero ? throw new ObjectDisposedException(nameof(WhisperContext)) : handle;
        public static WhisperContext Open(string model, bool dtw = false, int dtwPreset = 1, int dtwTop = 2)
        {
            ArgumentNullException.ThrowIfNull(model);
            if (dtw && !((dtwPreset == 1 && dtwTop > 0) || (dtwPreset >= 3 && dtwPreset <= 14))) throw new ArgumentException("DTW needs positive top-head count or model preset (3..14)");
            var p = NativeMethods.whisper_context_default_params_by_ref(); if (p == IntPtr.Zero) throw new InvalidOperationException("Whisper context parameters unavailable");
            try { if (dtw) NativeMethods.crispasr_ctx_params_set_dtw(p, true, dtwPreset, dtwTop); var ctx = NativeMethods.whisper_init_from_file_with_params_by_ref(model, p); if (ctx == IntPtr.Zero) throw new InvalidOperationException("Whisper model could not open"); return new WhisperContext(ctx); }
            finally { NativeMethods.whisper_free_context_params(p); }
        }
        private static void Set(IntPtr p, int? value, Action<IntPtr, int> setter) { if (value.HasValue) setter(p, value.Value); }
        private static void Flag(IntPtr p, bool? value, Action<IntPtr, int> setter) { if (value.HasValue) setter(p, value.Value ? 1 : 0); }
        public WhisperSegment[] Transcribe(float[] pcm, WhisperDecodeOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(pcm); options ??= new WhisperDecodeOptions(); if (pcm.Length == 0 || options.Processors < 1 || options.Strategy < 0 || options.Strategy > 1) throw new ArgumentException("Nonempty PCM, positive processor count and greedy/beam strategy required");
            lock (gate)
            {
                var ctx = Handle; using var strings = new Utf8Pointers(new[] { options.Language ?? "", options.InitialPrompt ?? "", options.VadModelPath ?? "" }); var p = NativeMethods.whisper_full_default_params_by_ref(options.Strategy); if (p == IntPtr.Zero) throw new InvalidOperationException("Whisper decode parameters unavailable");
                try
                {
                    if (options.Language != null) NativeMethods.crispasr_params_set_language(p, Marshal.ReadIntPtr(strings.Pointer, 0));
                    if (options.InitialPrompt != null) NativeMethods.crispasr_params_set_initial_prompt(p, Marshal.ReadIntPtr(strings.Pointer, IntPtr.Size));
                    if (options.VadModelPath != null) NativeMethods.crispasr_params_set_vad_model_path(p, Marshal.ReadIntPtr(strings.Pointer, IntPtr.Size * 2));
                    if (options.Silent) { NativeMethods.crispasr_params_set_print_realtime(p, 0); NativeMethods.crispasr_params_set_print_progress(p, 0); NativeMethods.crispasr_params_set_print_timestamps(p, 0); NativeMethods.crispasr_params_set_print_special(p, 0); }
                    Flag(p, options.Translate, NativeMethods.crispasr_params_set_translate);
                    Flag(p, options.DetectLanguage, NativeMethods.crispasr_params_set_detect_language);
                    Flag(p, options.TokenTimestamps, NativeMethods.crispasr_params_set_token_timestamps);
                    Flag(p, options.SplitOnWord, NativeMethods.crispasr_params_set_split_on_word);
                    Flag(p, options.NoContext, NativeMethods.crispasr_params_set_no_context);
                    Flag(p, options.SingleSegment, NativeMethods.crispasr_params_set_single_segment);
                    Flag(p, options.PrintRealtime, NativeMethods.crispasr_params_set_print_realtime);
                    Flag(p, options.PrintProgress, NativeMethods.crispasr_params_set_print_progress);
                    Flag(p, options.PrintTimestamps, NativeMethods.crispasr_params_set_print_timestamps);
                    Flag(p, options.PrintSpecial, NativeMethods.crispasr_params_set_print_special);
                    Flag(p, options.SuppressBlank, NativeMethods.crispasr_params_set_suppress_blank);
                    Flag(p, options.Vad, NativeMethods.crispasr_params_set_vad);
                    Flag(p, options.Tdrz, NativeMethods.crispasr_params_set_tdrz);
                    Set(p, options.NThreads, NativeMethods.crispasr_params_set_n_threads);
                    Set(p, options.MaxLen, NativeMethods.crispasr_params_set_max_len);
                    Set(p, options.BestOf, NativeMethods.crispasr_params_set_best_of);
                    Set(p, options.MaxTokens, NativeMethods.crispasr_params_set_max_tokens);
                    Set(p, options.AltN, NativeMethods.crispasr_params_set_alt_n);
                    Set(p, options.VadMinSpeechMs, NativeMethods.crispasr_params_set_vad_min_speech_ms);
                    Set(p, options.VadMinSilenceMs, NativeMethods.crispasr_params_set_vad_min_silence_ms);
                    if (options.Temperature.HasValue) NativeMethods.crispasr_params_set_temperature(p, options.Temperature.Value);
                    if (options.VadThreshold.HasValue) NativeMethods.crispasr_params_set_vad_threshold(p, options.VadThreshold.Value);
                    int rc = options.Processors == 1 ? NativeMethods.whisper_full_by_ref(ctx, p, pcm, pcm.Length) : NativeMethods.crispasr_transcribe_parallel_by_ref(ctx, p, pcm, pcm.Length, options.Processors);
                    if (rc != 0) throw new InvalidOperationException($"Whisper transcription failed (rc={rc})"); return ReadSegments(ctx);
                }
                finally { NativeMethods.whisper_free_params(p); }
            }
        }
        private static WhisperSegment[] ReadSegments(IntPtr ctx)
        {
            int count = NativeMethods.whisper_full_n_segments(ctx); if (count < 0) throw new InvalidOperationException("Invalid segment count"); var segments = new WhisperSegment[count]; var scratch = new byte[256];
            for (int i = 0; i < count; ++i)
            {
                int n = NativeMethods.whisper_full_n_tokens(ctx, i); if (n < 0) throw new InvalidOperationException("Invalid token count"); var tokens = new WhisperToken[n];
                for (int j = 0; j < n; ++j)
                {
                    var raw = NativeMethods.CopyUtf8Bytes(NativeMethods.whisper_full_get_token_text(ctx, i, j)); long t0 = NativeMethods.crispasr_token_t0(ctx, i, j), t1 = NativeMethods.crispasr_token_t1(ctx, i, j); bool timed = t0 >= 0 && t1 >= t0; int na = NativeMethods.crispasr_token_n_alts(ctx, i, j); if (na < 0) throw new InvalidOperationException("Invalid alternative count"); var alternatives = new WhisperAlternative[na];
                    for (int k = 0; k < na; ++k) { int length; while ((length = NativeMethods.crispasr_token_alt_text(ctx, i, j, k, scratch, scratch.Length)) < 0) { if (scratch.Length >= 1048576) throw new InvalidOperationException("Alternative text too large"); scratch = new byte[checked(scratch.Length * 2)]; } if (length >= scratch.Length) throw new InvalidOperationException("Invalid alternative byte count"); var bytes = new byte[length]; Array.Copy(scratch, bytes, length); alternatives[k] = new WhisperAlternative(NativeMethods.crispasr_token_alt_id(ctx, i, j, k), Encoding.UTF8.GetString(bytes), NativeMethods.crispasr_token_alt_p(ctx, i, j, k), bytes); }
                    tokens[j] = new WhisperToken(NativeMethods.whisper_full_get_token_id(ctx, i, j), Encoding.UTF8.GetString(raw), raw, timed ? Session.Seconds(t0) : -1, timed ? Session.Seconds(t1) : -1, NativeMethods.crispasr_token_p(ctx, i, j), Session.Seconds(NativeMethods.crispasr_token_dtw_t(ctx, i, j)), alternatives);
                }
                segments[i] = new WhisperSegment(NativeMethods.PtrToUtf8(NativeMethods.whisper_full_get_segment_text(ctx, i)) ?? "", Session.Seconds(NativeMethods.whisper_full_get_segment_t0(ctx, i)), Session.Seconds(NativeMethods.whisper_full_get_segment_t1(ctx, i)), NativeMethods.whisper_full_get_segment_no_speech_prob(ctx, i), tokens);
            }
            return segments;
        }
        public LanguageDetection DetectLanguage(float[] pcm, int threads = 4) { ArgumentNullException.ThrowIfNull(pcm); lock (gate) { var buffer = new byte[64]; float p = NativeMethods.crispasr_detect_language(Handle, pcm, pcm.Length, threads, buffer, buffer.Length); if (p < 0) throw new InvalidOperationException($"Whisper language detection failed (rc={p})"); return new LanguageDetection(NativeMethods.NullTerminated(buffer), p); } }
        public StreamDecoder StreamOpen(int threads = 4, int stepMs = 3000, int lengthMs = 10000, int keepMs = 200, string language = "en", bool translate = false)
        {
            lock (gate) { var native = NativeMethods.crispasr_stream_open(Handle, threads, stepMs, lengthMs, keepMs, language, translate ? 1 : 0); if (native == IntPtr.Zero) throw new InvalidOperationException("Whisper stream could not open"); var stream = new StreamDecoder(() => { _ = Handle; }, s => streams.Remove(s), native); streams.Add(stream); return stream; }
        }
        public void Dispose() { lock (gate) { if (handle != IntPtr.Zero) { foreach (var stream in new List<StreamDecoder>(streams)) stream.Dispose(); NativeMethods.whisper_free(handle); handle = IntPtr.Zero; } } }
    }
}
