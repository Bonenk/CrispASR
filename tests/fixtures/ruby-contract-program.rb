require ARGV.fetch(0)
s = Whisper::CrispASR::Session
c = Whisper::CrispASR
chat = Whisper::CrispASR::Chat
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
# Original 29 assertions (preserve exact semantics)
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
# New: audio load family
al=c.audio_load('probe.wav')
eq(al[:pcm].size,3);eq(al[:sample_rate],16000);eq(al[:n_samples],3)
al2=c.audio_load_at_rate('probe.wav', 24000)
eq(al2[:sample_rate],24000)
st=c.audio_load_stereo('probe.wav')
eq(st[:channels],2);eq(st[:left].size,3);eq(st[:right].size,3)
eq(st[:sample_rate],16000)
eq(c.audio_resample([0.1,0.2,0.3], 16000, 24000).size,3)
eq(c.audio_resample([], 16000, 24000).size,0)
eq(c.pcm_to_wav([0.1,0.2,0.3], 16000).bytesize,4)
eq(c.pcm_to_wav([0.1,0.2,0.3], 16000).encoding, Encoding::ASCII_8BIT)
cs=c.c2pa_sign("W\0AV", "audio/wav")
eq(cs.bytesize,4)
c.test_audio_mode(3)
eq(c.audio_load_stereo('probe')[:channels],1)
eq(c.test_audio_live,0)
c.test_audio_mode(2)
reject(RuntimeError){c.audio_load('probe')}
reject(RuntimeError){c.audio_load_at_rate('probe',24000)}
reject(RuntimeError){c.audio_load_stereo('probe')}
reject(RuntimeError){c.audio_resample([1,3,5],16000,24000)}
eq(c.test_audio_live,0)
c.test_audio_mode(0)
eq(s.kokoro_lang_is_german('de'),true);eq(s.kokoro_lang_is_german('en'),false)
eq(s.kokoro_lang_has_native_voice('en'),true);eq(s.kokoro_lang_has_native_voice('xx'),false)
c.watermark_load_model('probe.gguf')
eq(c.c_api_version, 'probe')
eq(c.dart_helpers_version, 'probe-helpers')
c.set_gpu_backend('cuda')
eq(c.backend_caps('foo'), 'asr,tts')
eq(c.backend_caps_list.include?('foo'), true)
eq(c.detect_backends_from_gguf('probe.gguf').include?('foo'), true)
eq(s.backend(h), 'probe')
sc=s.score_texts(h,[1,3,5],'en',nil,['hello','hi'])
eq(sc[:logprobs][0], -0.0);eq(sc[:n_tokens][1], 2)
s.set_pcm_sample_rate(h, 24000)
s.set_parakeet_att_context(h, 10, 20)
s.set_grammar_strict(h, true)
s.set_grammar_strict(h, false)
# truecase/pcs/punc
punc=s.punc_init('model')
eq(s.punc_process(punc,'hello'), 'Hello!')
s.punc_free(punc)
tc=s.truecase_init('model')
eq(s.truecase_process(tc,'hello'), "Gr\u00FC\u00DFe")
s.truecase_free(tc)
pcs=s.pcs_init('model')
eq(s.pcs_process(pcs,'hello'), 'Hello.')
s.pcs_free(pcs)
c.free_aligner_cache
# streamed polling
eq(c.get_progress, 42)
c.reset_progress
eq(c.get_progress, -1)
eq(c.get_streamed_segment_count, 2)
segs=c.drain_streamed_segments
eq(segs.first[:text], 'Japanese')
eq(segs.first[:t0], 123)
c.reset_streamed_segments
eq(c.get_streamed_segment_count, 0)
eq(c.get_streamed_token_count, 3)
toks=c.drain_streamed_tokens
eq(toks.size, 4)
eq(toks.first.bytes, [0xe6,0x97])
c.reset_streamed_tokens
eq(c.get_streamed_token_count, 0)
# callbacks with rooted lifetime and exception deferral
progress=[]
s.set_progress_callback(h, ->(pr,t){ progress << [pr,t] })
seg_texts=[]
s.set_segment_callback(h, ->(txt,t0,t1,idx){ seg_texts << txt })
tok_texts=[]
s.set_token_callback(h, ->(txt,idx){ tok_texts << txt })
s.transcribe(h,[1,3,5])
eq(progress.first, [3,6])
eq(seg_texts.first, "日本語")
eq(tok_texts.first.bytes, [0xe6,0x97])
s.set_segment_callback(h, ->(txt,t0,t1,idx){ raise ArgumentError,'cb err' })
reject(ArgumentError){ s.transcribe(h,[1,3,5]) }
s.set_segment_callback(h, ->(txt,t0,t1,idx){ s.transcribe(h,[1,3,5]) })
reject(RuntimeError){ s.transcribe(h,[1,3,5]) }
s.set_progress_callback(h, ->(_,_){raise ArgumentError,'root sentinel'})
s.set_segment_callback(h, ->(*_){GC.compact if GC.respond_to?(:compact)})
begin
  s.transcribe(h,[1,3,5]);raise 'missing deferred callback error'
rescue ArgumentError => e
  eq(e.message,'root sentinel')
end
s.set_progress_callback(h, nil)
s.set_segment_callback(h, nil)
s.set_token_callback(h, nil)
s.transcribe(h,[1,3,5])
# cross-session / unknown-userdata routing — verifies exact userdata dispatch, no fallback scan

# Use distinct fake handles to avoid fixture returning same pointer (1) for every open
h1=0x1001; h2=0x1002
prog1=[]; prog2=[]
s.set_progress_callback(h1, ->(a,b){ prog1 << [a,b] })
s.set_progress_callback(h2, ->(a,b){ prog2 << [a,b] })
c.test_invoke_progress(h1)
eq(prog1.size,1); eq(prog2.size,0)
eq(prog1.first, [3,6])
c.test_invoke_progress(h2)
eq(prog2.size,1); eq(prog1.size,1)
# unknown/null must not route to other session (no fallback scan)
c.test_invoke_progress(nil)
eq(prog1.size,1); eq(prog2.size,1)
c.test_invoke_progress(0xdeadbeef)
eq(prog1.size,1); eq(prog2.size,1)
seg1=[]; seg2=[]
s.set_segment_callback(h1, ->(txt,t0,t1,idx){ seg1 << txt })
s.set_segment_callback(h2, ->(txt,t0,t1,idx){ seg2 << txt })
c.test_invoke_segment(h1)
eq(seg1.first, "日本語"); eq(seg2.size,0)
c.test_invoke_segment(h2)
eq(seg2.first, "日本語"); eq(seg1.size,1)
c.test_invoke_segment(nil)
eq(seg1.size,1); eq(seg2.size,1)
c.test_invoke_segment(0xdeadbeef)
eq(seg1.size,1); eq(seg2.size,1)
tok1=[]; tok2=[]
s.set_token_callback(h1, ->(txt,idx){ tok1 << txt })
s.set_token_callback(h2, ->(txt,idx){ tok2 << txt })
c.test_invoke_token(h1)
eq(tok1.first.bytes, [0xe6,0x97]); eq(tok2.size,0)
c.test_invoke_token(h2)
eq(tok2.first.bytes, [0xe6,0x97]); eq(tok1.size,1)
c.test_invoke_token(nil)
eq(tok1.size,1); eq(tok2.size,1)
c.test_invoke_token(0xdeadbeef)
eq(tok1.size,1); eq(tok2.size,1)
# rooted callbacks clear on close — delete hash entry, no dangling proc
s.close(h1)
c.test_invoke_progress(h1)
eq(prog1.size,1)
c.test_invoke_segment(h1)
eq(seg1.size,1)
c.test_invoke_token(h1)
eq(tok1.size,1)
s.close(h2)
c.test_invoke_progress(h2)
eq(prog2.size,1)
# pitch / separate / dimension getters
pit=s.pitch(h,[1,3,5],10.0)
eq(pit[:n_frames], 3)
eq(pit[:frames].size, 9)
eq(pit[:sample_rate], 16000)
eq(s.pitch_n_frames(h), 3)
eq(s.pitch_sample_rate(h), 16000)
eq(s.pitch_frame(h,0)[:f0_hz], 440.0)
sep=s.separate(h,[0,0,0,0])
eq(sep[:n_stems], 2)
eq(sep[:stems].first[:name], 'vocals')
eq(sep[:sample_rate], 44100)
eq(s.separate_n_stems(h), 2)
eq(s.separate_sample_rate(h), 44100)
eq(s.convert_sample_rate(h), 40000)
eq(s.piano_n_notes(h), 2)
eq(s.chords_n_spans(h), 2)
eq(s.beats_n_events(h), 2)
eq(s.tab_n_frames(h), 2)
# transcribe_vad plain (without lang) with full metadata
vad_plain=s.transcribe_vad(h,[1,3,5],'vad')
eq(vad_plain.first[:tokens].size,2)
eq(vad_plain.first[:has_timestamps], true)
# registry / cache
rl=c.registry_lookup('backend')
eq(rl[:filename], 'model.gguf')
rbf=c.registry_lookup_by_filename('model.gguf')
eq(rbf[:url], 'https://example/model.gguf')
rbi=c.registry_default_bundle_info('backend')
eq(rbi[:license], 'MIT')
rba=c.registry_default_bundle_artifact('backend',0)
eq(rba[:kind], 0)
eq(c.cache_ensure_file('f','https://example'), '/tmp/cache/model.gguf')
eq(c.cache_dir(nil), '/tmp/cache')
opts={method:4,n_threads:7,slice_t0_cs:125,model_path:'model',embedder_path:'embed'}
diarized=c.diarize_turns([1,3,5],nil,[{t0:100,t1:200}],opts)
eq(diarized[:segments].first[:speaker],1);eq(diarized[:turns].first[:t0],125);eq(diarized[:turns].first[:t1],250);eq(diarized[:turns].first[:speaker],2)
reject(ArgumentError){c.diarize_turns([1,3,5],[1],[{t0:100,t1:200}],opts)}
# chat full lifecycle with structured copying and no fake streaming
ch=chat.open('probe', {'n_ctx'=>2048,'n_gpu_layers'=>0,'use_mmap'=>false,'chat_template'=>'gemma'})
eq(chat.template_name(ch), 'probe-template')
eq(chat.n_ctx(ch), 2048)
eq(chat.count_tokens(ch, [{'role'=>'user','content'=>'hi'}]), 2)
eq(chat.memory_estimate('probe',{}) > 0, true)
eq(chat.ai_disclosure_text.include?('AI'), true)
msg=[{'role'=>'system','content'=>'system'},{'role'=>'user','content'=>"日本語"}]
eq(chat.generate(ch, msg, {'max_tokens'=>7,'seed'=>0xffffffff, 'temperature'=>0.8,'stop'=>['STOP']}), "Gr\u00FC\u00DFe")
eq(chat.generate(ch,msg,{prefill_only:true}), "")
chunks2=[]
chat.generate_stream(ch, msg, {}){|chunk| GC.compact if GC.respond_to?(:compact); chunks2 << chunk }
eq(chunks2.size, 2)
eq(chunks2.first.bytes, [0xe6,0x97])
eq(chunks2.last, "\xA5")
# chat token/abort exception + errorcode 40 survival
begin
  chat.generate_stream(ch, msg, {}){|chunk| raise ArgumentError,'tok boom' }
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'tok boom')
end
chat.set_abort_callback(ch, ->{ raise ArgumentError,'abort boom' })
begin
  chat.generate(ch, msg, {})
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'abort boom')
end
first_abort=0;second_abort=0
other_chat=chat.open('probe',{})
chat.set_abort_callback(ch, ->{first_abort+=1;GC.compact if GC.respond_to?(:compact);true})
chat.set_abort_callback(other_chat, ->{second_abort+=1;true})
GC.compact if GC.respond_to?(:compact)
eq(chat.generate(ch,msg,{}),"Grüße");eq(first_abort,1);eq(second_abort,0)
eq(chat.generate(other_chat,msg,{}),"Grüße");eq(first_abort,1);eq(second_abort,1)
chat.close(other_chat)
chat.set_abort_callback(ch, nil)
begin
  chat.generate_stream(ch, msg, {}){|chunk| raise ArgumentError,'tok2' }
  raise "should fail"
rescue ArgumentError => e
  eq(e.message, 'tok2')
end
aborted=false
chat.set_abort_callback(ch, ->{ aborted=true; false })
begin
  chat.generate(ch, msg, {})
rescue RuntimeError => e
  eq(e.message.include?('cancelled'), true)
end
eq(aborted, true)
chat.set_abort_callback(ch, nil)
chat.reset(ch)
chat.close(ch)
begin
  chat.generate(ch, msg, {})
  raise "should fail"
rescue RuntimeError, StandardError
  $count += 1
end
reject(RuntimeError){ chat.open('fail', {}) }
pk=s.parakeet_init('probe',4,true)
pkr=s.parakeet_transcribe(pk,[1,3,5],123)
eq(pkr[:text],'日本語');eq(pkr[:words].first[1],223);eq(pkr[:tokens].first[1],-1);eq(pkr[:tokens].map{|t|t[0]}.join.force_encoding('UTF-8'),'日')
reject(ArgumentError){s.parakeet_transcribe(pk,[1,3,5],'de')}
s.parakeet_free(pk)
titan=s.titanet_init('probe',4)
eq(s.titanet_embed(titan,[1,3,5]).length,192);eq(s.titanet_embed(titan,[1,3,5]).first,1.0)
[2,3,4].each{|mode|c.test_titanet_mode(mode);reject(RuntimeError){s.titanet_embed(titan,[1,3,5])}};c.test_titanet_mode(0)
reject(ArgumentError){s.titanet_embed(titan,[])};s.titanet_free(titan);s.titanet_free(titan)
eq(c.test_titanet_live,0);reject(RuntimeError){s.titanet_embed(titan,[1,3,5])}
s.parakeet_free(pk);reject(RuntimeError){s.parakeet_transcribe(pk,[1,3,5],0)}
s.close(h)
puts "PASS Ruby ABI: #{$count} assertions"
