"""Real JNA calls through the Java wrapper into public-header C probes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
JAR=os.environ.get('CRISPASR_JNA_JAR')
JAVA=os.environ.get('JAVA') or shutil.which('java')
PROGRAM=r'''
import io.github.ggerganov.whispercpp.*;
import com.sun.jna.*;
import java.util.*;
import java.nio.charset.StandardCharsets;
public class WrapperAbiProbe {
    public interface Probe extends Library {
        Probe INSTANCE=Native.load("crispasr",Probe.class);
        void probe_mode(int n);void probe_mutate();int probe_live();int probe_invalid_frees();int probe_result_frees();int probe_calls();void probe_utility_mode(int mode);int probe_utility_live();
    }
    static int assertions;
    static void eq(Object a,Object b){if(!Objects.equals(a,b))throw new AssertionError(a+" != "+b);assertions++;}
    static void reject(Class<? extends Throwable> cls,Runnable fn) {
        try {fn.run();}catch(Throwable t){if(cls.isInstance(t)){assertions++;return;}throw t;}
        throw new AssertionError("expected "+cls.getName());
    }
    public static void main(String[] args) {
        float[] pcm={1,3,5};boolean old=args.length>0;
        try(CrispasrSession s=CrispasrSession.open("probe",4)) {
            CrispasrSession.Segment a=s.transcribe(pcm)[0];
            eq(a.t0,123L);eq(a.words[0].t0,-1L);eq(a.words[0].hasTimestamps(),false);
            eq(a.words[0].alts[0].text,"alternative");eq(a.speaker,"speaker");
            if(old){eq(a.tokens.length,0);System.out.println("PASS Java older ABI: "+assertions);return;}
            eq(a.tokens.length,2);eq(a.tokens[0].rawText[0],(byte)0xe6);
            eq(a.tokens[0].hasTimestamps(),false);eq(a.tokens[1].t0,123L);
            eq(new String(new byte[]{a.tokens[0].rawText[0],a.tokens[0].rawText[1],a.tokens[1].rawText[0]},StandardCharsets.UTF_8),"日");
            eq(s.transcribeChunked(pcm,0,-1,null)[0].tokens.length,2);
            eq(s.transcribeVad(pcm,16000,"vad")[0].tokens.length,2);
            eq(s.transcribeWithLogits(pcm,null).segments[0].tokens.length,2);
            CrispasrSessionTasks.PianoResult piano=s.piano(pcm);eq(piano.programs[1],128);eq(piano.sampleRate,16000);
            eq(s.chords(pcm,16000).names[1],"Am");eq(s.beats(pcm).tempoBpm,123.5f);
            eq(s.backend(),"probe");eq(CrispasrSession.loadAudio("x").samples[2],.75f);
            eq(CrispasrSession.loadAudio("x",24000).sampleRate,24000);
            eq(CrispasrSession.loadAudioStereo("x").samples.length,6);
            Probe.INSTANCE.probe_utility_mode(3);eq(CrispasrSession.loadAudioStereo("x").sourceChannels,1);
            eq(CrispasrSession.resample(new float[0],16000,24000).length,0);
            eq(CrispasrSession.resample(pcm,16000,24000)[0],.25f);
            eq(CrispasrSession.pcmToWav(pcm,16000)[1],(byte)0);
            eq(Arrays.toString(CrispasrSession.signC2pa(new byte[]{1,0,2},"audio/wav",null,null)),"[1, 0, 2]");
            Probe.INSTANCE.probe_utility_mode(2);
            reject(IllegalStateException.class,()->CrispasrSession.loadAudio("x"));
            reject(IllegalStateException.class,()->CrispasrSession.loadAudioStereo("x"));
            reject(IllegalStateException.class,()->CrispasrSession.signC2pa(new byte[]{1},"audio/wav",null,null));
            eq(Probe.INSTANCE.probe_utility_live(),0);Probe.INSTANCE.probe_utility_mode(0);
            eq(s.scoreTexts(pcm,new String[]{"one","two"},null,null).tokenCounts[1],3);
            eq(CrispasrSession.cApiVersion(),"probe");eq(CrispasrSession.helperVersion(),"probe-helpers");
            eq(CrispasrSession.backendCapabilities("foo")[0],"asr");
            try(CrispasrUtilities.TextProcessor processor=new CrispasrUtilities.TextProcessor("probe",false)){eq(processor.process("test"),"Grüße");}
            try(CrispasrUtilities.TextProcessor processor=new CrispasrUtilities.TextProcessor("probe",true)){eq(processor.process("test"),"Hello.");}
            eq(s.tab(pcm,22050).emissions.length,12);eq(s.tab(pcm,22050).openMidi[1],45);
            Probe.INSTANCE.probe_mutate();eq(piano.notes[0],0f);
            eq(s.convert(new float[]{0,1,2,3,4,5},pcm,0).length,6);
            reject(IllegalArgumentException.class,()->s.convert(new float[6],pcm,0,new float[5],null));
            Probe.INSTANCE.probe_mode(1);
            eq(s.piano(pcm).notes.length,0);eq(s.chords(pcm,16000).spans.length,0);eq(s.beats(pcm).events.length,0);
            Probe.INSTANCE.probe_mode(0);
            s.setVoiceSamples(pcm,24000,"hello");s.setTtsCfgScale(1.5f);s.tadaSetMakerefModels(null,null);
            reject(IllegalStateException.class,()->s.setTtsCfgScale(2));
            eq(s.getDisclaimerPcm()[0],.25f);eq(Probe.INSTANCE.probe_live(),0);
            List<float[]> chunks=new ArrayList<>();s.synthesizeStreaming("two. chunks.",(x,last)->chunks.add(x));
            eq(chunks.size(),2);eq(chunks.get(0)[0],.25f);eq(chunks.get(1)[0],.75f);
            reject(IllegalStateException.class,()->s.synthesizeStreaming("test",(x,last)->s.close()));
            int before=Probe.INSTANCE.probe_result_frees();
            s.setTokenCallback((x,i)->{throw new IllegalArgumentException("sentinel");});
            reject(IllegalArgumentException.class,()->s.transcribe(pcm));
            eq(Probe.INSTANCE.probe_result_frees(),before+1);s.setTokenCallback(null);
            eq(Probe.INSTANCE.probe_live(),0);eq(Probe.INSTANCE.probe_invalid_frees(),0);
        }
        CrispasrSession owner=CrispasrSession.open("probe",1);
        CrispasrSession.Stream stream=owner.streamOpen(100,300,10,null,false);
        eq(stream.feed(pcm),1);stream.setLiveDecode(true);eq(stream.getText().text,"ok");
        int calls=Probe.INSTANCE.probe_calls();owner.close();eq(Probe.INSTANCE.probe_calls(),calls+2);
        reject(IllegalStateException.class,()->stream.feed(pcm));stream.close();
        System.out.println("PASS Java ABI: "+assertions+" assertions");
    }
}
'''


@unittest.skipUnless(JAVA and JAR and Path(JAR).is_file(), 'Java runtime/compiler and CRISPASR_JNA_JAR required')
class JavaWrapperContract(unittest.TestCase):
    def test_transport_ownership_metadata_and_task_operations(self):
        with tempfile.TemporaryDirectory(prefix='java-contract-') as tmp:
            path=Path(tmp);(path/'WrapperAbiProbe.java').write_text(PROGRAM)
            sources=[ROOT/'bindings/java/src/main/java/io/github/ggerganov/whispercpp'/name
                     for name in ('CrispasrSession.java','CrispasrSessionTasks.java','CrispasrUtilities.java')]
            javac=shutil.which('javac')
            compiler=[javac] if javac else [JAVA,'-m','jdk.compiler/com.sun.tools.javac.Main']
            subprocess.run(compiler+['-encoding','UTF-8','-cp',JAR,'-d',str(path),*map(str,sources),str(path/'WrapperAbiProbe.java')],check=True)
            for old in (False,True):
                directory=path/('old' if old else 'current');directory.mkdir()
                subprocess.run([os.environ.get('CC','cc'),'-std=c11','-shared','-fPIC','-Werror',
                                '-I'+str(ROOT/'include'), '-I'+str(ROOT/'ggml/include'), '-DCRISPASR_PROBE_EXISTING_STREAM_OPS', *(['-DCRISPASR_PROBE_OLD_TOKENS'] if old else []),
                                str(ROOT/'tests/fixtures/python-wrapper-abi.c'),str(ROOT/'tests/fixtures/wrapper-session-metadata.c'),str(ROOT/'tests/fixtures/wrapper-stream-lifetime.c'),str(ROOT/'tests/fixtures/wrapper-audio-utilities.c'),
                                '-o',str(directory/'libcrispasr.so')],check=True)
                run=subprocess.run([JAVA,'-Djna.tmpdir='+str(path),'-Djna.library.path='+str(directory),
                                    '-cp',str(path)+os.pathsep+JAR,'WrapperAbiProbe',*(['old'] if old else [])],capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr);self.assertIn('PASS Java',run.stdout)
                print(run.stdout.strip())


if __name__=='__main__':unittest.main()
