using System;
using System.Runtime.InteropServices;

namespace CrispASR
{
    /// <summary>Settings for the shared VAD transcription pipeline. Durations are milliseconds.</summary>
    [StructLayout(LayoutKind.Sequential)]
    public sealed class VadOptions
    {
        public float Threshold = 0.5f;
        public int MinSpeechDurationMs = 250;
        public int MinSilenceDurationMs = 100;
        public int SpeechPadMs = 30;
        /// <summary>Maximum chunk length in seconds; zero disables splitting.</summary>
        public int ChunkSeconds = 30;
        public int NThreads = 4;

        internal IntPtr Allocate()
        {
            if (!float.IsFinite(Threshold) || Threshold < 0 || Threshold > 1)
                throw new ArgumentOutOfRangeException(nameof(Threshold));
            if (MinSpeechDurationMs < 0 || MinSilenceDurationMs < 0 || SpeechPadMs < 0 || ChunkSeconds < 0 || NThreads < 1)
                throw new ArgumentOutOfRangeException(nameof(VadOptions), "Durations must be nonnegative and threads positive");
            var ptr = Marshal.AllocHGlobal(Marshal.SizeOf<VadOptions>());
            try { Marshal.StructureToPtr(this, ptr, false); return ptr; }
            catch { Marshal.FreeHGlobal(ptr); throw; }
        }
    }
}
