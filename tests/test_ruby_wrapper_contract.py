"""Compile the actual Ruby extension against public-header, model-free probes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUBY = os.environ.get('CRISPASR_RUBY') or shutil.which('ruby')
PROGRAM = r'''
require ARGV.fetch(0)
s = Whisper::CrispASR::Session
$count=0
def eq(a,b)
  raise "#{a.inspect} != #{b.inspect}" unless a == b
  $count += 1
end
def reject(type)
  begin
    yield
  rescue type
    $count += 1
    return
  end
  raise "expected #{type}"
end
h=s.open('probe',1)
a=s.transcribe(h,[1,3,5]).first
eq(a[:speaker],'speaker');eq(a[:words].first[:alts].first[:text],'alternative')
eq(a[:tokens].size,2);eq(a[:tokens].map{|x|x[:raw_text]}.join.force_encoding('UTF-8'),'日')
eq(a[:tokens].first[:raw_text].bytes,[0xe6,0x97]);eq(a[:tokens].first[:text].valid_encoding?,true)
eq(a[:tokens].first[:t0],-1);eq(a[:tokens].first[:has_timestamps],false)
eq(a[:t0],123)
eq(s.transcribe_chunked(h,[1,3,5],0,-1,nil).first[:tokens].size,2)
eq(s.transcribe_with_logits(h,[1,3,5]).first.first[:tokens].size,2)
b=s.transcribe_vad_lang(h,[1,3,5],'vad',nil).first
eq(b[:tokens].size,2);eq(b[:tokens].first[:t0],-1.0);eq(b[:t0],1.23)
piano=s.piano(h,[1,3,5]);eq(piano[:programs],[24,128]);eq(piano[:notes].size,8)
eq(s.chords(h,[1,3,5],16000)[:names],['C','Am'])
eq(s.beats(h,[1,3,5])[:tempo_bpm],123.5)
eq(s.tab(h,[1,3,5],22050)[:emissions].size,12)
eq(s.convert(h,[0,1,2,3,4,5],[1,3,5],0),[0,1,2,3,4,5])
reject(ArgumentError){s.convert(h,[0,1,2,3,4,5],[1,3,5],0,[0]*5,nil)}
s.set_voice_samples(h,[1,3,5],24000,'hello');s.set_tts_cfg_scale(h,1.5)
reject(RuntimeError){s.set_tts_cfg_scale(h,2)}
s.tada_set_makeref_models(h,nil,nil)
eq(s.get_disclaimer_pcm(h).first,0.25)
chunks=[];s.synthesize_streaming(h,'a. b.'){|pcm,last|chunks << [pcm,last]}
eq(chunks.size,2);eq(chunks.first.first.first,0.25);eq(chunks.last.first.first,0.75);eq(chunks.last.last,true)
reject(RuntimeError){s.synthesize_streaming(h,'test'){|pcm,last|s.close(h)}}
reject(ArgumentError){s.synthesize_streaming(h,'test'){|pcm,last|raise ArgumentError,'sentinel'}}
s.close(h)
puts "PASS Ruby ABI: #{$count} assertions"
'''


@unittest.skipUnless(RUBY and shutil.which('cc'), 'Ruby SDK and C compiler required')
class RubyWrapperContract(unittest.TestCase):
    def test_metadata_music_conversion_and_stream_callbacks(self):
        env = os.environ.copy()
        # SDK/header paths may be overridden for a relocated development toolchain.
        config = subprocess.check_output([RUBY, '-rrbconfig', '-e',
            'puts RbConfig::CONFIG.values_at("rubyhdrdir","rubyarchhdrdir")'], env=env, text=True).splitlines()
        headers = os.environ.get('CRISPASR_RUBY_HEADERS')
        if headers:
            config = headers.split(os.pathsep)
        with tempfile.TemporaryDirectory(prefix='ruby-wrapper-abi-') as tmp:
            tmp = Path(tmp)
            flags = ['-fPIC', *['-I'+h for h in config], '-I'+str(ROOT/'include')]
            obj = tmp/'wrapper.o'
            subprocess.run(['cc', *flags, '-c', str(ROOT/'bindings/ruby/ext/ruby_crispasr_session.c'), '-o', str(obj)], check=True)
            probe = tmp/'probe.o'
            metadata = tmp/'metadata.o'
            for target, source in [(probe,'python-wrapper-abi.c'),(metadata,'wrapper-session-metadata.c')]:
                subprocess.run(['cc','-fPIC','-I'+str(ROOT/'include'),'-c',str(ROOT/'tests/fixtures'/source),'-o',str(target)],check=True)
            def symbols(args):
                return {line.split()[-1] for line in subprocess.check_output(['nm',*args],text=True).splitlines() if line.split()}
            needed = symbols(['-u',str(obj)])
            supplied = symbols(['-g','--defined-only',str(probe),str(metadata)])
            # Unexercised legacy registration dependencies abort if accidentally called.
            # Every exercised function is supplied by a header-checked concrete probe.
            unused = sorted(n for n in needed-supplied if n.startswith('crispasr_'))
            stub = tmp/'unused.c'
            stub.write_text('#include <stdlib.h>\n'+''.join('void '+n+'(void){abort();}\n' for n in unused))
            init = tmp/'init.c'
            init.write_text('#include <ruby.h>\nvoid init_ruby_crispasr_session(VALUE*);\nvoid Init_wrapper_probe(void){VALUE m=rb_define_module("Whisper");init_ruby_crispasr_session(&m);}\n')
            lib = tmp/'wrapper_probe.so'
            subprocess.run(['cc','-shared',*flags,str(obj),str(probe),str(metadata),str(stub),str(init),'-lpthread','-o',str(lib)],check=True)
            program = tmp/'probe.rb';program.write_text(PROGRAM)
            result = subprocess.run([RUBY,str(program),str(lib)],text=True,capture_output=True,env=env)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('PASS Ruby ABI:',result.stdout)
            print(result.stdout.strip())

if __name__ == '__main__':
    unittest.main()
