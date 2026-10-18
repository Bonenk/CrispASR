using System;
using System.Runtime.InteropServices;
using System.Text;
using CrispASR;
class WhisperProbe {
    [DllImport("crispasr")]static extern int probe_legacy_live();
    [DllImport("crispasr")]static extern int probe_legacy_contexts();
    [DllImport("crispasr")]static extern int probe_legacy_processors();
    [DllImport("crispasr")]static extern int probe_legacy_dtw();
    [DllImport("crispasr")]static extern int probe_legacy_progress_zero();
    [DllImport("crispasr")]static extern void probe_legacy_mode(int m);
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(){
        Reject<ArgumentException>(()=>WhisperContext.Open("probe",true,2));Eq(probe_legacy_live(),0);
        Reject<InvalidOperationException>(()=>WhisperContext.Open("fail"));Eq(probe_legacy_live(),0);
        var ctx=WhisperContext.Open("probe",true);Eq(probe_legacy_live(),0);Eq(probe_legacy_contexts(),1);Eq(probe_legacy_dtw(),102);
        var pcm=new float[]{1,3,5};var options=new WhisperDecodeOptions{Language="de",InitialPrompt="Grüße",TokenTimestamps=true,Processors=2};
        var result=ctx.Transcribe(pcm,options);Eq(probe_legacy_processors(),2);Eq(probe_legacy_progress_zero(),1);Eq(probe_legacy_live(),0);
        Eq(result[0].Text,"日");Eq(result[0].Start,1.0);Eq(result[0].End,2.0);Eq(result[0].NoSpeechProbability,.125f);
        var token=result[0].Tokens[0];Eq(token.HasTimestamps,false);Eq(token.Start,-1.0);Eq(token.End,-1.0);Eq(token.DtwTime,1.4);Eq(token.Id,10);Eq(token.Probability,.5f);Eq(token.RawText.Length,2);Eq(token.Text.Contains('�'),true);
        Eq(result[0].Tokens[1].Start,1.23);Eq(result[0].Tokens[1].End,1.5);Eq(token.Alternatives[0].Id,42);Eq(token.Alternatives[0].Text,"alternative");Eq(token.Alternatives[0].Probability,.25f);
        var joined=new byte[3];Array.Copy(token.RawText,joined,2);Array.Copy(result[0].Tokens[1].RawText,0,joined,2,1);Eq(Encoding.UTF8.GetString(joined),"日");
        probe_legacy_mode(1);Eq(ctx.Transcribe(pcm,options).Length,0);probe_legacy_mode(2);Reject<InvalidOperationException>(()=>ctx.Transcribe(pcm,options));Eq(probe_legacy_live(),0);
        probe_legacy_mode(3);Reject<InvalidOperationException>(()=>ctx.Transcribe(pcm,options));Eq(probe_legacy_live(),0);probe_legacy_mode(0);
        ctx.Dispose();ctx.Dispose();Eq(probe_legacy_contexts(),0);Reject<ObjectDisposedException>(()=>ctx.Transcribe(pcm,options));
        Console.WriteLine($"PASS C# legacy Whisper: {checks} assertions");
    }
}
