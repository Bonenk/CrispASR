using System;
using System.Runtime.InteropServices;
using System.Text;
using CrispASR;
class ParakeetProbe {
    [DllImport("crispasr")]static extern int probe_parakeet_contexts();
    [DllImport("crispasr")]static extern int probe_parakeet_results();
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(){
        Reject<InvalidOperationException>(()=>new ParakeetContext("fail"));var ctx=new ParakeetContext("probe");Eq(probe_parakeet_contexts(),1);
        var pcm=new float[]{1,3,5};var result=ctx.Transcribe(pcm,1.23);Eq(result.Text,"日本語");Eq(result.Words[0].T0,2.23);Eq(result.Words[0].T1,3.23);Eq(result.Words[0].P,-1f);Eq(result.Tokens[0].T0,-1.0);Eq(result.Tokens[0].P,.75f);Eq(result.Tokens[1].T0,2.46);Eq(result.Tokens[0].RawText[0],(byte)0xe6);Eq(probe_parakeet_results(),0);
        Reject<ArgumentException>(()=>ctx.Transcribe(Array.Empty<float>()));Reject<ArgumentException>(()=>ctx.Transcribe(pcm,double.NaN));Reject<InvalidOperationException>(()=>ctx.Transcribe(new float[]{2,4,6}));Eq(probe_parakeet_results(),0);
        ctx.Dispose();ctx.Dispose();Eq(probe_parakeet_contexts(),0);Reject<ObjectDisposedException>(()=>ctx.Transcribe(pcm));Eq(result.Words[0].Text,"日本語");Console.WriteLine($"PASS C# legacy Parakeet: {checks} assertions");
    }
}
