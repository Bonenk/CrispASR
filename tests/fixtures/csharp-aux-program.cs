using System;
using System.Runtime.InteropServices;
using CrispASR;
class SessionAuxProbe {
    [DllImport("crispasr")]static extern void probe_aux_mode(int mode);
    [DllImport("crispasr")]static extern int probe_aux_live();
    [DllImport("crispasr")]static extern int probe_aux_segment_size();
    [DllImport("crispasr")]static extern int probe_aux_options_size();
    static int checks;
    static void Eq<T>(T a,T b){if(!object.Equals(a,b))throw new Exception($"{a} != {b}");checks++;}
    static void Reject<T>(Action f)where T:Exception{try{f();}catch(T){checks++;return;}throw new Exception("Expected "+typeof(T));}
    static void Main(){
        Eq(Marshal.SizeOf<NativeMethods.NativeDiarizationSegment>(),probe_aux_segment_size());Eq(Marshal.SizeOf<NativeMethods.NativeDiarizationOptions>(),probe_aux_options_size());
        using(var session=Session.Open("probe",new SessionOptions{Threads=7,UseGpu=false,Verbosity=2,FlashAttention=false,GpuLayers=3})){Eq(session.Backend,"probe");}
        var pcm=new float[]{1,3,5};var spans=Session.VadSlices("vad",pcm);Eq(spans[0].T0,1.25);Eq(spans[0].T1,2.5);
        probe_aux_mode(1);Eq(Session.VadSlices("vad",pcm).Length,0);probe_aux_mode(2);Reject<InvalidOperationException>(()=>Session.VadSlices("vad",pcm));Eq(probe_aux_live(),0);probe_aux_mode(0);
        Eq(Session.DetectTextLanguage("test","model").Code,"de");Eq(Session.DetectTextLanguage("test","model").Probability,.875f);probe_aux_mode(2);Reject<InvalidOperationException>(()=>Session.DetectTextLanguage("test","model"));probe_aux_mode(0);
        using(var embedder=new SpeakerEmbedder("probe")){Eq(embedder.Dimension,3);Eq(embedder.Name,"embed-probe");Eq(embedder.Embed(pcm)[2],.75f);probe_aux_mode(2);Reject<InvalidOperationException>(()=>embedder.Embed(pcm));probe_aux_mode(0);foreach(int invalid in new[]{15,16,17}){probe_aux_mode(invalid);Reject<InvalidOperationException>(()=>embedder.Embed(pcm));}probe_aux_mode(0);}
        Eq(SpeakerEmbedder.Cluster(new float[]{1,0,0,0,1,0},3).ClusterCount,2);Reject<ArgumentException>(()=>SpeakerEmbedder.Cluster(new float[5],3));
        foreach(int invalid in new[]{11,12,13}) {probe_aux_mode(invalid);Reject<InvalidOperationException>(()=>SpeakerEmbedder.Cluster(new float[]{1,0,0,0,1,0},3));}
        probe_aux_mode(14);var unassigned=SpeakerEmbedder.Cluster(new float[]{1,0,0,0,1,0},3);Eq(unassigned.ClusterCount,0);Eq(unassigned.Labels[1],-1);probe_aux_mode(0);
        Reject<ArgumentException>(()=>SpeakerEmbedder.Cluster(new float[]{1,0,0,0,1,0},3,float.NaN));
        Reject<ArgumentException>(()=>new SpeakerEmbedder("probe\0bad"));
        Reject<ArgumentException>(()=>new SpeakerEmbedder("probe",cacheDirectory:"cache\0bad"));
        Reject<ArgumentException>(()=>new PyannoteCache(pcm,"model\0bad"));
        Reject<ArgumentException>(()=>new PyannoteCache(Array.Empty<float>(),"model"));
        var segments=new[]{new DiarizationSegment(1,2)};var options=new DiarizationOptions{Method=4,Threads=7,SliceOffsetSeconds=1.25,Model="model",Embedder="embed",MinSpeakers=1,MaxSpeakers=3,NumSpeakers=2};
        var diarized=Diarization.Assign(pcm,segments,options);Eq(diarized.Segments[0].Speaker,1);Eq(diarized.Turns[0].Start,1.25);Eq(diarized.Turns[0].End,2.5);Eq(segments[0].Speaker,-1);
        Eq(Diarization.Assign(pcm,segments,options,includeTurns:false).Turns.Length,0);Reject<ArgumentException>(()=>Diarization.Assign(pcm,segments,options,new float[1]));
        using(var cache=new PyannoteCache(pcm,"model")){var result=cache.Apply(segments,2.5);Eq(result[0].Speaker,2);Eq(result[0].Start,1.0);}
        Eq(ModelRegistry.Lookup("foo")!.Value.Filename,"model.gguf");Eq(ModelRegistry.Lookup("model.gguf",true)!.Value.Url,"https://test/model");Eq(ModelRegistry.Lookup("missing"),null);
        Eq(ModelRegistry.Backends()[1],"bar");Eq(ModelRegistry.DefaultBundle("foo")!.Value.Artifacts[0].Kind,1);Eq(ModelRegistry.CacheDirectory(),"cache");var entry=ModelRegistry.Lookup("foo")!.Value;Eq(ModelRegistry.EnsureFile(entry),"cache/model.gguf");
        probe_aux_mode(2);Reject<InvalidOperationException>(()=>ModelRegistry.CacheDirectory());Reject<InvalidOperationException>(()=>ModelRegistry.EnsureFile(entry));probe_aux_mode(0);
        Eq(TextUtilities.KokoroLanguageIsGerman("de"),true);Eq(TextUtilities.KokoroLanguageIsGerman("en"),false);Eq(TextUtilities.KokoroLanguageHasNativeVoice("en"),true);Eq(TextUtilities.KokoroLanguageHasNativeVoice("xx"),false);Eq(TextUtilities.LcsDedupPrefixCount(new[]{1,2},new[]{2,3}),1);
        Eq(probe_aux_live(),0);Console.WriteLine($"PASS C# auxiliary: {checks} assertions");
    }
}
