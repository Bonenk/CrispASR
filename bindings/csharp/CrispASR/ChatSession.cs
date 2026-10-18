using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace CrispASR
{
    internal static class ChatNative
    {
        private const string Lib = "crispasr";
        [StructLayout(LayoutKind.Sequential)]
        internal struct OpenParams
        {
            internal int Threads, BatchThreads, Context, Batch, MicroBatch, GpuLayers;
            [MarshalAs(UnmanagedType.I1)] internal bool Mmap;
            [MarshalAs(UnmanagedType.I1)] internal bool Mlock;
            [MarshalAs(UnmanagedType.I1)] internal bool Embeddings;
            internal IntPtr Template;
        }
        [StructLayout(LayoutKind.Sequential)]
        internal struct GenerateParams
        {
            internal int MaxTokens; internal float Temperature; internal int TopK;
            internal float TopP, MinP, RepeatPenalty; internal int RepeatLastN; internal uint Seed;
            internal IntPtr Stop; internal UIntPtr StopCount;
            [MarshalAs(UnmanagedType.I1)] internal bool PrefillOnly;
        }
        [StructLayout(LayoutKind.Sequential)]
        internal struct Error
        {
            internal int Code;
            [MarshalAs(UnmanagedType.ByValArray, SizeConst = 256)] internal byte[] Message;
            internal static Error Empty => new Error { Message = new byte[256] };
        }
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate void TokenCallback(IntPtr text, IntPtr user);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)] internal delegate bool AbortCallback(IntPtr user);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_chat_open_params_default(out OpenParams value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_chat_generate_params_default(out GenerateParams value);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_chat_open([MarshalAs(UnmanagedType.LPUTF8Str)] string path, ref OpenParams options, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_chat_close(IntPtr session);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_chat_reset(IntPtr session, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_chat_generate(IntPtr session, IntPtr messages, UIntPtr n, ref GenerateParams options, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_chat_generate_stream(IntPtr session, IntPtr messages, UIntPtr n, ref GenerateParams options, TokenCallback callback, IntPtr user, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_chat_set_abort_callback(IntPtr session, AbortCallback? callback, IntPtr user);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_chat_template_name(IntPtr session);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_chat_n_ctx(IntPtr session);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_chat_count_tokens(IntPtr session, IntPtr messages, UIntPtr n, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern UIntPtr crispasr_chat_memory_estimate([MarshalAs(UnmanagedType.LPUTF8Str)] string path, ref OpenParams options, ref Error error);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_chat_string_free(IntPtr text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_chat_ai_disclosure_text();
    }
    public sealed class ChatOptions
    {
        public int? Threads { get; set; }
        public int? BatchThreads { get; set; }
        public int? ContextSize { get; set; }
        public int? BatchSize { get; set; }
        public int? MicroBatchSize { get; set; }
        public int? GpuLayers { get; set; }
        public bool? UseMmap { get; set; }
        public bool? UseMlock { get; set; }
        public string? Template { get; set; }
        internal ChatNative.OpenParams Native()
        {
            ChatNative.crispasr_chat_open_params_default(out var p);
            p.Threads = Threads ?? p.Threads; p.BatchThreads = BatchThreads ?? p.BatchThreads;
            p.Context = ContextSize ?? p.Context; p.Batch = BatchSize ?? p.Batch; p.MicroBatch = MicroBatchSize ?? p.MicroBatch;
            p.GpuLayers = GpuLayers ?? p.GpuLayers; p.Mmap = UseMmap ?? p.Mmap; p.Mlock = UseMlock ?? p.Mlock;
            p.Embeddings = false; p.Template = Template == null ? IntPtr.Zero : Marshal.StringToCoTaskMemUTF8(Template); return p;
        }
    }
    public sealed class ChatGenerateOptions
    {
        public int? MaxTokens { get; set; }
        public float? Temperature { get; set; }
        public int? TopK { get; set; }
        public float? TopP { get; set; }
        public float? MinP { get; set; }
        public float? RepeatPenalty { get; set; }
        public int? RepeatLastN { get; set; }
        public uint? Seed { get; set; }
        public string[] Stop { get; set; } = Array.Empty<string>();
        public bool PrefillOnly { get; set; }
        internal ChatNative.GenerateParams Native(Utf8Pointers stops)
        {
            ChatNative.crispasr_chat_generate_params_default(out var p);
            p.MaxTokens = MaxTokens ?? p.MaxTokens; p.Temperature = Temperature ?? p.Temperature; p.TopK = TopK ?? p.TopK;
            p.TopP = TopP ?? p.TopP; p.MinP = MinP ?? p.MinP; p.RepeatPenalty = RepeatPenalty ?? p.RepeatPenalty;
            p.RepeatLastN = RepeatLastN ?? p.RepeatLastN; p.Seed = Seed ?? p.Seed;
            p.Stop = stops.Pointer; p.StopCount = (UIntPtr)stops.Count; p.PrefillOnly = PrefillOnly; return p;
        }
    }
    public readonly struct ChatMessage
    {
        public string Role { get; }
        public string Content { get; }
        public ChatMessage(string role, string content) { ArgumentNullException.ThrowIfNull(role); ArgumentNullException.ThrowIfNull(content); Role = role; Content = content; }
    }
    public sealed class ChatException : InvalidOperationException
    {
        public int NativeCode { get; }
        internal ChatException(int code, string message) : base(message) { NativeCode = code; }
    }
    // Owns a contiguous pointer array plus UTF-8 allocations. Used for both
    // const char* stop arrays and the two-pointer message records.
    internal sealed class Utf8Pointers : IDisposable
    {
        internal IntPtr Pointer { get; private set; }
        internal int Count { get; }
        private readonly IntPtr[] strings;
        private bool disposed;
        internal Utf8Pointers(IReadOnlyList<string> values)
        {
            ArgumentNullException.ThrowIfNull(values); Count = values.Count; strings = new IntPtr[Count];
            Pointer = Count == 0 ? IntPtr.Zero : Marshal.AllocHGlobal(checked(Count * IntPtr.Size));
            try { for (int i = 0; i < Count; ++i) { ArgumentNullException.ThrowIfNull(values[i]); if (values[i].Contains('\0')) throw new ArgumentException("Embedded NUL in UTF-8 argument"); strings[i] = Marshal.StringToCoTaskMemUTF8(values[i]); Marshal.WriteIntPtr(Pointer, i * IntPtr.Size, strings[i]); } }
            catch { Dispose(); throw; }
        }
        internal static Utf8Pointers Messages(IReadOnlyList<ChatMessage> messages)
        {
            ArgumentNullException.ThrowIfNull(messages); var values = new string[checked(messages.Count * 2)];
            for (int i = 0; i < messages.Count; ++i) { values[i * 2] = messages[i].Role; values[i * 2 + 1] = messages[i].Content; }
            return new Utf8Pointers(values);
        }
        public void Dispose() { if (disposed) return; disposed = true; foreach (var p in strings) if (p != IntPtr.Zero) Marshal.FreeCoTaskMem(p); if (Pointer != IntPtr.Zero) { Marshal.FreeHGlobal(Pointer); Pointer = IntPtr.Zero; } }
    }
    /// <summary>Owned text-chat session. Pass the complete conversation on every generation to reuse its KV prefix.</summary>
    public sealed class ChatSession : IDisposable
    {
        private IntPtr handle;
        private readonly object gate = new();
        private volatile bool inCallback;
        private ExceptionDispatchInfo? callbackError;
        private Func<bool>? abort;
        private readonly ChatNative.AbortCallback abortNative;
        private CancellationToken cancellation;
        private ChatSession(IntPtr value) { handle = value; abortNative = Abort; }
        private void Check() { if (inCallback) throw new InvalidOperationException("Do not re-enter/close chat from a callback"); if (handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(ChatSession)); }
        private static void Error(int rc, ChatNative.Error error) { if (rc == 0 && error.Code == 0) return; if (rc == 40 || error.Code == 40) throw new OperationCanceledException("Chat generation cancelled"); throw new ChatException(error.Code != 0 ? error.Code : rc, NativeMethods.NullTerminated(error.Message)); }
        private void Capture(Exception error) { Interlocked.CompareExchange(ref callbackError, ExceptionDispatchInfo.Capture(error), null); }
        private void ThrowCallback() { var error = Interlocked.Exchange(ref callbackError, null); error?.Throw(); }
        private bool Abort(IntPtr user)
        {
            if (cancellation.IsCancellationRequested || Volatile.Read(ref callbackError) != null) return false;
            var cb = abort; if (cb == null) return true; inCallback = true;
            try { return cb(); } catch (Exception e) { Capture(e); return false; } finally { inCallback = false; }
        }
        public static ChatSession Open(string model, ChatOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(model); var p = (options ?? new ChatOptions()).Native(); var error = ChatNative.Error.Empty;
            try { var h = ChatNative.crispasr_chat_open(model, ref p, ref error); if (h == IntPtr.Zero) { Error(-1, error); throw new ChatException(-1, "Chat open failed"); } return new ChatSession(h); }
            finally { if (p.Template != IntPtr.Zero) Marshal.FreeCoTaskMem(p.Template); }
        }
        public static string AiDisclosureText => NativeMethods.PtrToUtf8(ChatNative.crispasr_chat_ai_disclosure_text()) ?? "";
        public static ulong MemoryEstimate(string model, ChatOptions? options = null)
        {
            ArgumentNullException.ThrowIfNull(model); var p = (options ?? new ChatOptions()).Native(); var error = ChatNative.Error.Empty;
            try { var size = ChatNative.crispasr_chat_memory_estimate(model, ref p, ref error); Error(0, error); return size.ToUInt64(); }
            finally { if (p.Template != IntPtr.Zero) Marshal.FreeCoTaskMem(p.Template); }
        }
        public string TemplateName { get { Check(); lock (gate) { Check(); return NativeMethods.PtrToUtf8(ChatNative.crispasr_chat_template_name(handle)) ?? ""; } } }
        public int ContextSize { get { Check(); lock (gate) { Check(); return ChatNative.crispasr_chat_n_ctx(handle); } } }
        public void Reset() { Check(); lock (gate) { Check(); var e = ChatNative.Error.Empty; Error(ChatNative.crispasr_chat_reset(handle, ref e), e); } }
        public int CountTokens(IReadOnlyList<ChatMessage> messages)
        {
            Check(); lock (gate)
            {
                Check(); using var values = Utf8Pointers.Messages(messages); var e = ChatNative.Error.Empty;
                int count = ChatNative.crispasr_chat_count_tokens(handle, values.Pointer, (UIntPtr)messages.Count, ref e); Error(count < 0 ? count : 0, e); return count;
            }
        }
        /// <summary>Callback returns true to continue. Register before generation; callback must be fast and cannot re-enter this session.</summary>
        public void SetAbortCallback(Func<bool>? callback)
        {
            Check(); lock (gate) { Check(); abort = callback; ChatNative.crispasr_chat_set_abort_callback(handle, callback == null ? null : abortNative, IntPtr.Zero); }
        }
        public string Generate(IReadOnlyList<ChatMessage> messages, ChatGenerateOptions? options = null, CancellationToken cancellationToken = default)
        {
            Check(); lock (gate)
            {
                Check(); options ??= new ChatGenerateOptions(); using var values = Utf8Pointers.Messages(messages); using var stops = new Utf8Pointers(options.Stop);
                var p = options.Native(stops); var e = ChatNative.Error.Empty; cancellation = cancellationToken;
                ChatNative.crispasr_chat_set_abort_callback(handle, abortNative, IntPtr.Zero);
                var output = IntPtr.Zero;
                try { output = ChatNative.crispasr_chat_generate(handle, values.Pointer, (UIntPtr)messages.Count, ref p, ref e); ThrowCallback(); Error(output == IntPtr.Zero ? -1 : 0, e); return NativeMethods.PtrToUtf8(output) ?? ""; }
                finally { if (output != IntPtr.Zero) ChatNative.crispasr_chat_string_free(output); cancellation = default; GC.KeepAlive(abortNative); }
            }
        }
        /// <summary>Deliver owned raw UTF-8 chunks. Join bytes before decoding when pieces split a character. Callback errors propagate after native returns.</summary>
        public void GenerateStreaming(IReadOnlyList<ChatMessage> messages, Action<byte[]> onToken, ChatGenerateOptions? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(onToken); Check(); lock (gate)
            {
                Check(); options ??= new ChatGenerateOptions(); using var values = Utf8Pointers.Messages(messages); using var stops = new Utf8Pointers(options.Stop);
                var p = options.Native(stops); var e = ChatNative.Error.Empty; cancellation = cancellationToken;
                ChatNative.TokenCallback cb = (text, _) => { if (Volatile.Read(ref callbackError) != null) return; inCallback = true; try { onToken(NativeMethods.CopyUtf8Bytes(text)); } catch (Exception x) { Capture(x); } finally { inCallback = false; } };
                ChatNative.crispasr_chat_set_abort_callback(handle, abortNative, IntPtr.Zero);
                try { int rc = ChatNative.crispasr_chat_generate_stream(handle, values.Pointer, (UIntPtr)messages.Count, ref p, cb, IntPtr.Zero, ref e); ThrowCallback(); Error(rc, e); }
                finally { cancellation = default; GC.KeepAlive(cb); GC.KeepAlive(abortNative); }
            }
        }
        public void Dispose() { if (inCallback) throw new InvalidOperationException("Do not close chat from a callback"); lock (gate) { if (handle != IntPtr.Zero) { ChatNative.crispasr_chat_close(handle); handle = IntPtr.Zero; } GC.KeepAlive(abortNative); } }
    }
}
