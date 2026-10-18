using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace CrispASR
{
    internal static partial class NativeMethods
    {
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate void PcmCallback(IntPtr pcm, int n, int final, IntPtr data);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate void ProgressCallback(int processed, int total, IntPtr data);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate void SegmentCallback(IntPtr text, long t0, long t1, int index, IntPtr data);
        [UnmanagedFunctionPointer(CallingConvention.Cdecl)] internal delegate void TokenCallback(IntPtr text, int index, IntPtr data);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_voice_samples(IntPtr s, float[] pcm, int n, int rate, [MarshalAs(UnmanagedType.LPUTF8Str)] string? text);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_set_tts_cfg_scale(IntPtr s, float scale);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_tada_set_makeref_models(IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string? encoder, [MarshalAs(UnmanagedType.LPUTF8Str)] string? aligner);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_session_last_synth_error(IntPtr s);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_session_disclaimer_text();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_session_get_disclaimer_pcm(IntPtr s, out int n);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_session_synthesize_streaming(IntPtr s, [MarshalAs(UnmanagedType.LPUTF8Str)] string text, PcmCallback cb, IntPtr data);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_session_set_progress_callback(IntPtr s, ProgressCallback? cb, IntPtr data);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_session_set_segment_callback(IntPtr s, SegmentCallback? cb, IntPtr data);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_session_set_token_callback(IntPtr s, TokenCallback? cb, IntPtr data);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_get_progress();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_reset_progress();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_get_streamed_segment_count();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_drain_streamed_segments();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_reset_streamed_segments();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern int crispasr_get_streamed_token_count();
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern IntPtr crispasr_drain_streamed_tokens(out int n);
        [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)] internal static extern void crispasr_reset_streamed_tokens();
    }

    public sealed partial class Session
    {
        private bool _inCallback;
        private ExceptionDispatchInfo? _callbackError;
        private NativeMethods.ProgressCallback? _progressCallback;
        private NativeMethods.SegmentCallback? _segmentCallback;
        private NativeMethods.TokenCallback? _tokenCallback;

        private static void StrictCheck(int rc, string name)
        { if (rc != 0) throw new InvalidOperationException($"{name} failed (rc={rc})"); }
        private void InvokeCallback(Action action)
        {
            if (_callbackError != null) return;
            _inCallback = true;
            try { action(); }
            catch (Exception e) { _callbackError = ExceptionDispatchInfo.Capture(e); }
            finally { _inCallback = false; }
        }
        private void ThrowCallbackError()
        { var error=_callbackError; _callbackError=null; error?.Throw(); }
        private Segment[] ExtractSegmentsWithCallbacks(IntPtr result)
        { ThrowCallbackError(); return ExtractSegments(result); }

        /// <summary>Install mono reference PCM. Native consent and marking contracts apply.</summary>
        public void SetVoiceSamples(float[] pcm, int sampleRate, string? referenceText = null)
        {
            ArgumentNullException.ThrowIfNull(pcm);
            if (pcm.Length == 0 || sampleRate <= 0) throw new ArgumentException("Nonempty mono PCM and positive sample rate required");
            StrictCheck(NativeMethods.crispasr_session_set_voice_samples(Handle,pcm,pcm.Length,sampleRate,referenceText),"set_voice_samples");
        }
        public void SetTtsCfgScale(float scale) => StrictCheck(NativeMethods.crispasr_session_set_tts_cfg_scale(Handle,scale),"set_tts_cfg_scale");
        public void TadaSetMakerefModels(string? encoder = null, string? aligner = null)
            => StrictCheck(NativeMethods.crispasr_session_tada_set_makeref_models(Handle,encoder,aligner),"tada_set_makeref_models");
        public string LastSynthError => NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_last_synth_error(Handle)) ?? "";
        public static string DisclaimerText => NativeMethods.PtrToUtf8(NativeMethods.crispasr_session_disclaimer_text()) ?? "";
        /// <summary>Obtain neutral-voice disclosure PCM BEFORE installing a cloning voice.</summary>
        public float[] GetDisclaimerPcm()
        {
            IntPtr ptr=NativeMethods.crispasr_session_get_disclaimer_pcm(Handle,out int n);
            try
            {
                if(ptr==IntPtr.Zero || n<=0) throw new InvalidOperationException(LastSynthError);
                return CopyView(ptr,n);
            }
            finally { if(ptr!=IntPtr.Zero)NativeMethods.crispasr_pcm_free(ptr); }
        }
        /// <summary>Synchronous sentence chunks. PCM arrays are owned copies. Do not re-enter/dispose from callback.</summary>
        public void SynthesizeStreaming(string text, Action<float[],bool> callback)
        {
            ArgumentNullException.ThrowIfNull(text);ArgumentNullException.ThrowIfNull(callback);
            IntPtr handle=Handle;
            NativeMethods.PcmCallback cb=(pcm,n,final,_)=> {
                string error=(pcm==IntPtr.Zero || n<=0) ? LastSynthError : "";
                InvokeCallback(()=> {
                    if(pcm==IntPtr.Zero || n<=0)throw new InvalidOperationException(error);
                    callback(CopyView(pcm,n),final!=0);
                });
            };
            try
            {
                int rc=NativeMethods.crispasr_session_synthesize_streaming(handle,text,cb,IntPtr.Zero);
                ThrowCallbackError();StrictCheck(rc,"synthesize_streaming");
            }
            finally { GC.KeepAlive(cb); }
        }
        public void SetProgressCallback(Action<int,int>? callback)
        {
            var thunk=callback==null ? null : new NativeMethods.ProgressCallback((done,total,_)=>InvokeCallback(()=>callback(done,total)));
            NativeMethods.crispasr_session_set_progress_callback(Handle,thunk,IntPtr.Zero);_progressCallback=thunk;
        }
        /// <summary>Times are seconds; callback runs synchronously on the native inference thread.</summary>
        public void SetSegmentCallback(Action<string,double,double,int>? callback)
        {
            var thunk=callback==null ? null : new NativeMethods.SegmentCallback((text,t0,t1,index,_)=>InvokeCallback(()=>callback(NativeMethods.PtrToUtf8(text)??"",Seconds(t0),Seconds(t1),index)));
            NativeMethods.crispasr_session_set_segment_callback(Handle,thunk,IntPtr.Zero);_segmentCallback=thunk;
        }
        public void SetTokenCallback(Action<byte[],int>? callback)
        {
            var thunk=callback==null ? null : new NativeMethods.TokenCallback((text,index,_)=>InvokeCallback(()=>callback(NativeMethods.CopyUtf8Bytes(text),index)));
            NativeMethods.crispasr_session_set_token_callback(Handle,thunk,IntPtr.Zero);_tokenCallback=thunk;
        }
        public static int Progress => NativeMethods.crispasr_get_progress();
        public static void ResetProgress() => NativeMethods.crispasr_reset_progress();
        /// <summary>Global polling targets the last streaming session, which may differ from this instance.</summary>
        public static int StreamedSegmentCount => NativeMethods.crispasr_get_streamed_segment_count();
        public static int StreamedTokenCount => NativeMethods.crispasr_get_streamed_token_count();
        public static void ResetStreamedSegments() => NativeMethods.crispasr_reset_streamed_segments();
        public static void ResetStreamedTokens() => NativeMethods.crispasr_reset_streamed_tokens();
        public static Segment[] DrainStreamedSegments()
        {
            var result=NativeMethods.crispasr_drain_streamed_segments();
            if(result==IntPtr.Zero)return Array.Empty<Segment>();
            try { return ExtractSegments(result); }
            finally { NativeMethods.crispasr_session_result_free(result); }
        }
        public static byte[][] DrainStreamedTokens()
        {
            IntPtr pointer=NativeMethods.crispasr_drain_streamed_tokens(out int n);
            if(n<0 || (n>0 && pointer==IntPtr.Zero))throw new InvalidOperationException("invalid native token buffer");
            var result=new byte[n][];
            for(int i=0;i<n;i++) { result[i]=NativeMethods.CopyUtf8Bytes(pointer);pointer=IntPtr.Add(pointer,checked(result[i].Length+1)); }
            return result;
        }
    }
}
