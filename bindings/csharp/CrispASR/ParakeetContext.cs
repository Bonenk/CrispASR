using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    internal static partial class NativeMethods
    {
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_parakeet_init([MarshalAs(UnmanagedType.LPUTF8Str)] string model, int threads, int flash);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_parakeet_free(IntPtr ctx);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_parakeet_transcribe(IntPtr ctx, float[] pcm, int count, long offsetCs);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_parakeet_result_free(IntPtr result);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_parakeet_result_text(IntPtr r);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_parakeet_result_n_words(IntPtr r);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_parakeet_result_word_text(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_parakeet_result_word_t0(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_parakeet_result_word_t1(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_parakeet_result_n_tokens(IntPtr r);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_parakeet_result_token_text(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_parakeet_result_token_t0(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern long crispasr_parakeet_result_token_t1(IntPtr r, int i);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern float crispasr_parakeet_result_token_p(IntPtr r, int i);
    }
    public readonly record struct ParakeetTranscript(string Text, Word[] Words, SessionToken[] Tokens);
    /// <summary>Direct Parakeet API. Public times and offsets are seconds; no native language argument exists. One operation at a time.</summary>
    public sealed class ParakeetContext : IDisposable
    {
        private IntPtr handle;
        private readonly object gate = new();
        public ParakeetContext(string model, int threads = 4, bool flashAttention = true) { ArgumentNullException.ThrowIfNull(model); handle = NativeMethods.crispasr_parakeet_init(model, threads, flashAttention ? 1 : 0); if (handle == IntPtr.Zero) throw new InvalidOperationException("Parakeet model could not open"); }
        public ParakeetTranscript Transcribe(float[] pcm, double offsetSeconds = 0)
        {
            ArgumentNullException.ThrowIfNull(pcm); if (pcm.Length == 0 || !double.IsFinite(offsetSeconds)) throw new ArgumentException("Nonempty PCM and finite offset required");
            lock (gate)
            {
                if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(ParakeetContext));
                var r = NativeMethods.crispasr_parakeet_transcribe(handle, pcm, pcm.Length, checked((long)Math.Round(offsetSeconds * 100)));
                if (r == IntPtr.Zero) throw new InvalidOperationException("Parakeet transcription failed");
                try
                {
                    int nw = NativeMethods.crispasr_parakeet_result_n_words(r), nt = NativeMethods.crispasr_parakeet_result_n_tokens(r); if (nw < 0 || nt < 0) throw new InvalidOperationException("Invalid Parakeet result count"); var words = new Word[nw]; var tokens = new SessionToken[nt];
                    for (int i = 0; i < nw; ++i) words[i] = new Word(NativeMethods.PtrToUtf8(NativeMethods.crispasr_parakeet_result_word_text(r, i)) ?? "", Session.Seconds(NativeMethods.crispasr_parakeet_result_word_t0(r, i)), Session.Seconds(NativeMethods.crispasr_parakeet_result_word_t1(r, i)), -1);
                    for (int i = 0; i < nt; ++i) tokens[i] = new SessionToken(NativeMethods.CopyUtf8Bytes(NativeMethods.crispasr_parakeet_result_token_text(r, i)), Session.Seconds(NativeMethods.crispasr_parakeet_result_token_t0(r, i)), Session.Seconds(NativeMethods.crispasr_parakeet_result_token_t1(r, i)), NativeMethods.crispasr_parakeet_result_token_p(r, i));
                    return new ParakeetTranscript(NativeMethods.PtrToUtf8(NativeMethods.crispasr_parakeet_result_text(r)) ?? "", words, tokens);
                }
                finally { NativeMethods.crispasr_parakeet_result_free(r); }
            }
        }
        public void Dispose() { lock (gate) { if (handle != IntPtr.Zero) { NativeMethods.crispasr_parakeet_free(handle); handle = IntPtr.Zero; } } }
    }
}
