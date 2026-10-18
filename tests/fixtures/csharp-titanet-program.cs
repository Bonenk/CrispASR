using System;
using System.Runtime.InteropServices;
using CrispASR;
class TitaNetProbe {
    [DllImport("crispasr")]static extern int probe_titanet_live();
    [DllImport("crispasr")]static extern int probe_titanet_calls();
    [DllImport("crispasr")]static extern void probe_titanet_mode(int mode);
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(){
        using var ctx=TitaNet.Open("probe");var pcm=new float[]{1,3,5};var first=ctx.Embed(pcm);Eq(first.Length,192);Eq(first[0],1f);Eq(probe_titanet_live(),1);
        var second=ctx.Embed(pcm);second[0]=999;Eq(first[0],1f);
        probe_titanet_mode(2);Reject<InvalidOperationException>(()=>ctx.Embed(pcm));probe_titanet_mode(3);Reject<InvalidOperationException>(()=>ctx.Embed(pcm));probe_titanet_mode(4);Reject<InvalidOperationException>(()=>ctx.Embed(pcm));probe_titanet_mode(0);
        Reject<ArgumentException>(()=>ctx.Embed(Array.Empty<float>()));Reject<ArgumentNullException>(()=>ctx.Embed(null!));
        Eq(TitaNet.CosineSim(new float[]{1,0},new float[]{1,0}),1f);Reject<ArgumentException>(()=>TitaNet.CosineSim(new float[]{1},new float[2]));
        ctx.Dispose();ctx.Dispose();Eq(probe_titanet_live(),0);int calls=probe_titanet_calls();Reject<ObjectDisposedException>(()=>ctx.Embed(pcm));Eq(probe_titanet_calls(),calls);
        Console.WriteLine($"PASS C# TitaNet: {checks} assertions");
    }
}
