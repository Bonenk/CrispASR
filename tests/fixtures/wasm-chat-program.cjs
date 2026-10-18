const assert=require('node:assert/strict');
require(process.argv[2])().then(m=>{
 let n=0;const eq=(a,b)=>{assert.deepEqual(a,b);n++};const rejects=f=>{assert.throws(f);n++};
 eq(m.chatAvailable(),true);eq(m.chatAiDisclosure(),'You are talking to an AI.');eq(m.chatMemoryEstimate('x',{}),123);
 rejects(()=>new m.ChatSession('fail',{}));rejects(()=>new m.ChatSession('x',{embeddings:true}));eq(m.chatLiveSessions(),0);
 const s=new m.ChatSession('x',{chatTemplate:'gemma',useMmap:false,gpuLayers:0,contextSize:2048});eq(s.contextSize(),2048);eq(s.templateName(),'probe-template');
 const msgs=[{role:'system',content:'hello'},{role:'user',content:'日本語'}];
 eq(s.countTokens(msgs),9);const opts={maxTokens:7,seed:0xffffffff,stop:['STOP']};
 let r=s.generate(msgs,opts,null);eq(r.status,0);eq(r.text,'Grüße');eq(r.failed,false);eq(m.chatLiveStrings(),0);
 r=s.generate(msgs,{prefillOnly:true},null);eq(r.text,'');eq(r.status,0);eq(m.chatLiveStrings(),0);
 let chunks=[];r=s.generate(msgs,opts,p=>chunks.push(p));eq(r.status,0);eq(r.failed,false);eq(chunks.length,2);eq(Array.from(chunks[0]),[0xe6,0x97]);
 eq(new TextDecoder().decode(new Uint8Array([...chunks[0],...chunks[1]])),'日');
 const error=new Error('token');r=s.generate(msgs,opts,()=>{throw error});eq(r.failed,true);eq(r.error,error);eq(r.status,40);
 s.setAbortCallback(()=>false);r=s.generate(msgs,opts,null);eq(r.status,40);eq(r.failed,false);eq(m.chatLiveStrings(),0);
 const abortError=new Error('abort');s.setAbortCallback(()=>{throw abortError});r=s.generate(msgs,opts,null);eq(r.failed,true);eq(r.error,abortError);eq(r.status,40);
 s.setAbortCallback(null);r=s.generate(msgs,opts,()=>s.close());eq(r.failed,true);eq(r.error.message.includes('re-entry'),true);eq(m.chatLiveSessions(),1);
 s.reset();eq(m.chatResets(),1);s.close();s.close();rejects(()=>s.contextSize());s.delete();eq(m.chatLiveSessions(),0);
 const selfDeleting=new m.ChatSession('x',{});r=selfDeleting.generate(msgs,opts,()=>selfDeleting.delete());eq(r.failed,true);eq(m.chatLiveSessions(),0);
 // First callback may delete its Embind owner; the local shared state owns the
 // native context through the rest of the call. A second .delete() is caught.
 eq(m.chatLiveStrings(),0);console.log('PASS WASM chat: '+n+' assertions');
}).catch(e=>{console.error(e);process.exitCode=1});
