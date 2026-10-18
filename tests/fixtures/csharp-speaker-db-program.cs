using System;
using System.Runtime.InteropServices;
using CrispASR;
class Program {
    [DllImport("crispasr")] static extern void probe_db_mode(int mode);
    [DllImport("crispasr")] static extern int probe_db_live();
    [DllImport("crispasr")] static extern int probe_db_calls();
    static int checks;
    static void Eq<T>(T a,T b) { if(!Equals(a,b))throw new Exception($"{a} != {b}");++checks; }
    static void Reject<T>(Action f) where T:Exception {
        try{f();}catch(T){++checks;return;}throw new Exception("expected "+typeof(T));
    }
    static void Main() {
        var emb=new float[]{1,3,5};int calls=probe_db_calls();
        Reject<InvalidOperationException>(()=>SpeakerDb.Open("db","Alice,Bob",false));
        Reject<ArgumentException>(()=>SpeakerDb.Open("db","",true));Eq(probe_db_calls(),calls);
        using var db=SpeakerDb.Open("db","Alice,Bob",true);Eq(db.Count,2);Eq(probe_db_live(),1);
        var match=db.Match(emb);Eq(match.Name,"アリス");Eq(match.Score,.875f);
        probe_db_mode(1);Eq(db.Match(emb).Name,null);Eq(db.Match(emb).Score,-1f);
        Eq(db.Match(emb,-1f).Name,null);Eq(match.Name,"アリス");probe_db_mode(0);
        Reject<ArgumentException>(()=>db.Match(Array.Empty<float>()));Reject<ArgumentNullException>(()=>db.Match(null!));
        Reject<ArgumentOutOfRangeException>(()=>db.Match(emb,float.NaN));
        Reject<InvalidOperationException>(()=>SpeakerDb.Enroll("db","Alice",emb,false));
        Reject<ArgumentException>(()=>SpeakerDb.Enroll("db","Alice",Array.Empty<float>(),true));
        SpeakerDb.Enroll("db","Alice",emb,true);++checks;
        probe_db_mode(3);Reject<InvalidOperationException>(()=>SpeakerDb.Enroll("db","Alice",emb,true));
        probe_db_mode(2);Reject<InvalidOperationException>(()=>SpeakerDb.Open("db","Alice,Bob",true));probe_db_mode(0);
        db.Dispose();db.Dispose();Eq(probe_db_live(),0);calls=probe_db_calls();
        Reject<ObjectDisposedException>(()=>{var n=db.Count;});Reject<ObjectDisposedException>(()=>db.Match(emb));
        Eq(probe_db_calls(),calls);Console.WriteLine($"PASS C# speaker DB: {checks} assertions");
    }
}
