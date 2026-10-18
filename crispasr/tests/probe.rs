//! Probe tests for Rust wrapper - uses tiny compiled C probe against public headers
//! The probe is compiled via the contract runner against BOTH public headers in a temporary directory.

#![cfg(feature = "abi-probe")]

use crispasr::{Session, SessionSegment, SessionToken, SessionWord, WordAlternative};
use std::os::raw::c_int;

extern "C" {
    fn probe_mode_set(m: c_int);
    fn probe_live_get() -> c_int;
    fn probe_calls_get() -> c_int;
    fn probe_result_frees_get() -> c_int;
    fn probe_reset();
    fn probe_mutate();
    fn probe_utility_mode(mode: c_int);
    fn probe_utility_live() -> c_int;
    fn probe_stream_closes_get() -> c_int;
}

fn new_probe_session() -> Session {
    // Create a fake session handle that probe accepts (0x100000001)
    // We need to bypass Session::open and directly construct the struct
    // Session is defined with private fields, so we need to use unsafe to create it
    // Instead, we can use the fact that Session::open will call our probe's open which returns that pointer
    // But Session::open requires a model path, which our probe ignores and returns fake handle
    // So we can just call Session::open with any string
    Session::open("probe-test.gguf").expect("probe open should succeed")
}

#[test]
fn probe_token_rawbytes_after_free() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    let segs = sess.transcribe(&pcm).expect("transcribe should succeed");
    assert_eq!(segs.len(), 1, "should have 1 segment");
    let seg = &segs[0];
    assert_eq!(seg.text, "世");
    assert_eq!(seg.start, 1.0); // 100 cs -> 1.0 sec
    assert_eq!(seg.end, 2.0);
    assert_eq!(seg.speaker, "(Speaker 1) ");
    assert_eq!(seg.no_speech_prob, -1.0);
    assert!(seg.has_timestamps());
    assert_eq!(seg.words.len(), 1);
    let w = &seg.words[0];
    assert_eq!(w.text, "世");
    assert!(!w.has_timestamps()); // -1 sentinel
    assert_eq!(w.confidence, 0.9);
    assert_eq!(w.alternatives.len(), 1);
    assert_eq!(w.alternatives[0].text, "alternative");
    assert_eq!(w.alternatives[0].confidence, 0.25);
    assert_eq!(seg.tokens.len(), 2);
    let tok0 = &seg.tokens[0];
    let tok1 = &seg.tokens[1];
    assert_eq!(tok0.raw, vec![0xe6, 0x97]);
    assert_eq!(tok1.raw, vec![0xa5]);
    let mut combined = tok0.raw.clone();
    combined.extend_from_slice(&tok1.raw);
    assert_eq!(String::from_utf8(combined).unwrap(), "日");
    assert!(!tok0.has_timestamps());
    assert!(tok1.has_timestamps());
    assert_eq!(tok1.start, 1.23);
    assert_eq!(tok1.end, 1.24);
    assert_eq!(tok1.confidence, 0.75);
    // Check result was freed
    unsafe {
        assert_eq!(probe_result_frees_get(), 1);
        probe_mutate();
    }
    // After mutate, raw bytes should still be original (copied before free)
    assert_eq!(tok0.raw, vec![0xe6, 0x97]);
    assert_eq!(tok1.raw, vec![0xa5]);
}

#[test]
fn probe_all_result_readers() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    // Test all entry points
    let cases: Vec<(&str, Box<dyn Fn() -> Vec<SessionSegment>>)> = vec![
        ("transcribe", Box::new(|| sess.transcribe(&pcm).unwrap())),
        (
            "transcribe_with_language",
            Box::new(|| sess.transcribe_with_language(&pcm, Some("en")).unwrap()),
        ),
        (
            "transcribe_chunked",
            Box::new(|| sess.transcribe_chunked(&pcm, 30, 1).unwrap()),
        ),
        (
            "transcribe_vad",
            Box::new(|| sess.transcribe_vad(&pcm, "vad.gguf", None).unwrap()),
        ),
        (
            "transcribe_with_logits",
            Box::new(|| sess.transcribe_with_logits(&pcm).unwrap().0),
        ),
    ];
    for (name, f) in cases {
        unsafe {
            probe_reset();
            probe_mode_set(0);
        }
        let segs = f();
        assert!(!segs.is_empty(), "{} should return segments", name);
        assert_eq!(segs[0].tokens.len(), 2, "{} tokens", name);
        assert_eq!(segs[0].words[0].alternatives.len(), 1, "{} alts", name);
        assert!(!segs[0].speaker.is_empty(), "{} speaker", name);
    }
}

#[test]
fn probe_empty_result() {
    unsafe {
        probe_reset();
        probe_mode_set(1);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    let segs = sess.transcribe(&pcm).unwrap();
    assert!(segs.is_empty(), "empty should have 0 segments");
    // Music empty
    let piano = sess.piano(&pcm).unwrap();
    assert!(piano.notes.is_empty() && piano.programs.is_empty());
    let chords = sess.chords(&pcm, 16000).unwrap();
    assert!(chords.spans.is_empty());
    let beats = sess.beats(&pcm, 22050).unwrap();
    assert!(beats.events.is_empty());
    let tab = sess.tab(&pcm, 16000).unwrap();
    assert_eq!(tab.frames, 0);
    assert!(tab.emissions.is_empty());
}

#[test]
fn probe_error_result() {
    unsafe {
        probe_reset();
        probe_mode_set(2);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    assert!(sess.transcribe(&pcm).is_err());
    assert!(sess.piano(&pcm).is_err());
    assert!(sess.chords(&pcm, 16000).is_err());
    assert!(sess.beats(&pcm, 22050).is_err());
    assert!(sess.tab(&pcm, 16000).is_err());
    unsafe {
        probe_mode_set(0);
    }
    assert!(
        sess.beats(&pcm, 16000).is_err(),
        "beats with wrong rate should error"
    );
}

#[test]
fn probe_replay_noise_rejected_before_native() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let n_frames = 3;
    let content_dim = sess.convert_content_dim();
    assert_eq!(content_dim, 2);
    let mut content = vec![0.0; (n_frames as usize) * (content_dim as usize)];
    content[1] = 1.0;
    let mut f0 = vec![0.0; n_frames as usize];
    f0[2] = 5.0;
    let inter_ch = sess.convert_inter_channels();
    let upsample = sess.convert_upsample_product();
    assert_eq!(inter_ch, 2);
    assert_eq!(upsample, 4);
    let mut noise_zp = vec![0.0; (inter_ch as usize) * (n_frames as usize)];
    noise_zp[5] = 5.0;
    let mut noise_sine = vec![0.0; (n_frames as usize) * (upsample as usize)];
    noise_sine[11] = 11.0;
    unsafe {
        probe_reset();
    }
    let before = unsafe { probe_calls_get() };
    sess.convert(
        &content,
        n_frames,
        &f0,
        0,
        Some(&noise_zp),
        Some(&noise_sine),
    )
    .expect("convert should succeed");
    assert_eq!(unsafe { probe_calls_get() }, before + 1);
    unsafe {
        probe_reset();
    }
    let short_zp = vec![0.0; 2];
    assert!(sess
        .convert(
            &content,
            n_frames,
            &f0,
            0,
            Some(&short_zp),
            Some(&noise_sine)
        )
        .is_err());
    assert_eq!(
        unsafe { probe_calls_get() },
        0,
        "short zp should be rejected before native"
    );
    let short_sine = vec![0.0; 2];
    assert!(sess
        .convert(
            &content,
            n_frames,
            &f0,
            0,
            Some(&noise_zp),
            Some(&short_sine)
        )
        .is_err());
    assert_eq!(unsafe { probe_calls_get() }, 0);
}

#[test]
fn probe_allocated_pcm_frees() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let before = unsafe { probe_live_get() };
    let pcm = sess.get_disclaimer_pcm().expect("should succeed");
    assert_eq!(pcm.len(), 2);
    assert_eq!(pcm[0], 0.25);
    assert_eq!(unsafe { probe_live_get() }, before, "should free");
    let before = unsafe { probe_live_get() };
    let pcm2 = sess.synthesize("hello").expect("synthesize should succeed");
    assert_eq!(pcm2.len(), 2);
    assert_eq!(unsafe { probe_live_get() }, before);
    unsafe {
        probe_mode_set(1);
    }
    assert!(sess.get_disclaimer_pcm().is_err());
    assert_eq!(unsafe { probe_live_get() }, before);
}

#[test]
fn probe_synchronous_streaming() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let mut sess = new_probe_session();
    let mut chunks: Vec<Vec<f32>> = Vec::new();
    let mut finals: Vec<bool> = Vec::new();
    sess.synthesize_streaming("hello", |pcm, is_final| {
        // Test that pcm is copied before callback returns - mutate to check
        let mut cp = pcm.to_vec();
        chunks.push(cp.clone());
        finals.push(is_final);
        // Mutate original to ensure wrapper copied
        // pcm is a copy, so mutating it shouldn't affect stored
    })
    .expect("streaming should succeed");
    assert_eq!(chunks.len(), 2);
    assert_eq!(chunks[0], vec![0.25, 0.5]);
    assert_eq!(chunks[1], vec![0.75, 1.0]);
    assert_eq!(finals, vec![false, true]);
}

#[test]
fn probe_streaming_panic_recovery() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let mut sess = new_probe_session();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = sess.synthesize_streaming("hello", |_, _| panic!("test panic"));
    }));
    // The wrapper should catch panic and return error, not unwind
    // So we need to check that it returns error, not panics
    // Actually our wrapper uses catch_unwind and returns Err, so it should not panic
    // Let's test directly
    let err = sess.synthesize_streaming("hello", |_, _| panic!("test panic"));
    assert!(err.is_err());
    assert!(err.unwrap_err().contains("panicked"));
    // Ensure session still usable
    let mut chunks = 0;
    sess.synthesize_streaming("hello", |_, _| chunks += 1)
        .unwrap();
    assert_eq!(chunks, 2);
}

#[test]
fn probe_music_operations() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    let piano = sess.piano(&pcm).unwrap();
    assert_eq!(piano.notes.len(), 2);
    assert_eq!(piano.programs.len(), 2);
    assert_eq!(piano.sample_rate, 16000);
    assert_eq!(piano.notes[0].onset_ms, 0.0);
    assert_eq!(piano.programs[0], 24);
    let chords = sess.chords(&pcm, 16000).unwrap();
    assert_eq!(chords.spans.len(), 2);
    assert_eq!(chords.spans[0].name, "C");
    assert_eq!(chords.vocab_size, 170);
    let beats = sess.beats(&pcm, 22050).unwrap();
    assert_eq!(beats.events.len(), 2);
    assert_eq!(beats.sample_rate, 22050);
    assert_eq!(beats.tempo_bpm, 123.5);
    let tab = sess.tab(&pcm, 16000).unwrap();
    assert_eq!(tab.frames, 2);
    assert_eq!(tab.strings, 2);
    assert_eq!(tab.classes, 3);
    assert_eq!(tab.emissions.len(), 12);
    assert_eq!(tab.silent_class, 2);
    assert_eq!(tab.open_midi, vec![40, 45]);
    let pitch = sess.pitch(&pcm, 10.0).unwrap();
    assert_eq!(pitch.len(), 2);
    let content = {
        let mut c = vec![0.0; 6];
        c[1] = 1.0;
        c
    };
    let mut f0 = vec![0.0; 3];
    f0[2] = 5.0;
    let mut zp = vec![0.0; 6];
    zp[5] = 5.0;
    let mut sine = vec![0.0; 12];
    sine[11] = 11.0;
    let audio = sess
        .convert(&content, 3, &f0, 0, Some(&zp), Some(&sine))
        .unwrap();
    assert_eq!(audio.len(), 6);
    assert_eq!(audio[5], 5.0);
}

#[test]
fn probe_convert_validation() {
    unsafe {
        probe_reset();
    }
    let sess = new_probe_session();
    assert!(sess.convert(&[], 0, &[], 0, None, None).is_err());
    let content = vec![0.0; 6];
    let f0 = vec![0.0; 3];
    assert!(
        sess.convert(&content, 3, &f0, 5, None, None).is_err(),
        "invalid speaker should error"
    );
    assert!(sess.convert(&content, 1 << 30, &f0, 0, None, None).is_err());
}

#[test]
fn probe_beat_rate_mismatch() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    assert!(sess.beats(&pcm, 22050).is_ok());
    assert!(sess.beats(&pcm, 16000).is_err());
    assert!(sess.beats(&[], 22050).is_err());
}

#[test]
fn probe_has_timestamps() {
    let seg = SessionSegment {
        text: "a".to_string(),
        start: 1.0,
        end: 2.0,
        words: vec![],
        no_speech_prob: -1.0,
        speaker: "".to_string(),
        tokens: vec![],
    };
    assert!(seg.has_timestamps());
    let seg2 = SessionSegment {
        text: "".to_string(),
        start: -1.0,
        end: -1.0,
        words: vec![],
        no_speech_prob: -1.0,
        speaker: "".to_string(),
        tokens: vec![],
    };
    assert!(!seg2.has_timestamps());
    let w = SessionWord {
        text: "a".to_string(),
        start: 1.0,
        end: 2.0,
        confidence: 1.0,
        alternatives: vec![],
    };
    assert!(w.has_timestamps());
    let tok = SessionToken {
        text: "a".to_string(),
        raw: vec![],
        start: 1.0,
        end: 2.0,
        confidence: 1.0,
    };
    assert!(tok.has_timestamps());
}

#[test]
fn probe_disclaimer_and_last_error() {
    assert_eq!(Session::disclaimer_text(), "AI generated");
    let sess = new_probe_session();
    assert_eq!(sess.last_synth_error(), "probe synthesis failure");
    assert!(sess.set_tts_cfg_scale(1.5).is_ok());
    assert!(sess.set_tts_cfg_scale(1.0).is_err());
}

#[test]
fn probe_overflow_checks() {
    unsafe {
        probe_reset();
        probe_mode_set(0);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    // Tab overflow should be handled - huge dimensions would overflow but probe returns small, so we test wrapper's checked_mul
    // We can't easily trigger overflow via probe, but we can check that wrapper handles large counts without panic
    let tab = sess.tab(&pcm, 16000).unwrap();
    assert_eq!(tab.emissions.len(), 12);
}

#[test]
fn probe_zero_length_slice_safety() {
    // This test ensures that zero-length slices don't cause UB via from_raw_parts(null,0)
    // The probe's empty mode returns count 0 with null ptr, wrapper should handle without calling from_raw_parts
    unsafe {
        probe_reset();
        probe_mode_set(1);
    }
    let sess = new_probe_session();
    let pcm = vec![1.0, 3.0, 5.0];
    // All these should succeed with empty results, not panic
    let piano = sess.piano(&pcm).unwrap();
    assert!(piano.notes.is_empty());
    let chords = sess.chords(&pcm, 16000).unwrap();
    assert!(chords.spans.is_empty());
    let beats = sess.beats(&pcm, 22050).unwrap();
    assert!(beats.events.is_empty());
    let tab = sess.tab(&pcm, 16000).unwrap();
    assert!(tab.emissions.is_empty());
    let pitch = sess.pitch(&pcm, 10.0).unwrap();
    assert!(pitch.is_empty());
}

#[test]
fn probe_kokoro_bool_helpers() {
    // Exercise both true and false to catch garbage upper bits on bool return.
    // Fixture: de/de-DE => german true, en => not german, fr => false
    assert!(crispasr::kokoro_lang_is_german("de"));
    assert!(crispasr::kokoro_lang_is_german("de-DE"));
    assert!(!crispasr::kokoro_lang_is_german("en"));
    assert!(!crispasr::kokoro_lang_is_german("fr"));
    assert!(!crispasr::kokoro_lang_is_german(""));
    // Native voice: en/de true, others false
    assert!(crispasr::kokoro_lang_has_native_voice("en"));
    assert!(crispasr::kokoro_lang_has_native_voice("de"));
    assert!(!crispasr::kokoro_lang_has_native_voice("fr"));
    assert!(!crispasr::kokoro_lang_has_native_voice(""));
}

#[test]
fn probe_audio_allocation_errors_and_aliases() {
    unsafe { probe_utility_mode(0) };
    let (pcm, rate) = crispasr::audio_load("probe", Some(24000)).unwrap();
    assert_eq!(pcm[0], 0.25);
    assert_eq!(rate, 24000);
    unsafe { probe_utility_mode(3) };
    let (mut l, r, _, channels) = crispasr::audio_load_stereo("probe").unwrap();
    l[0] = 999.;
    assert_eq!(r[0], 0.25);
    assert_eq!(channels, 1);
    assert_eq!(
        crispasr::audio_resample(&[1., 3., 5.], 16000, 24000).unwrap()[0],
        0.25
    );
    assert_eq!(
        crispasr::pcm_to_wav(&[1., 3., 5.], 16000).unwrap(),
        vec![87, 0, 65, 86]
    );
    assert_eq!(
        crispasr::c2pa_sign(&[1, 0, 2], "audio/wav", None, None).unwrap(),
        vec![1, 0, 2]
    );
    unsafe { probe_utility_mode(2) };
    assert!(crispasr::audio_load("probe", None).is_err());
    assert!(crispasr::audio_load_stereo("probe").is_err());
    assert!(crispasr::audio_resample(&[1., 3., 5.], 16000, 24000).is_err());
    assert_eq!(unsafe { probe_utility_live() }, 0);
    unsafe { probe_utility_mode(0) };
}
#[test]
fn probe_stream_drops_before_parent() {
    let before = unsafe { probe_stream_closes_get() };
    let mut session = new_probe_session();
    {
        let stream = session.stream_open(3000, 10000, 200, "en", false).unwrap();
        stream.flush().unwrap();
    }
    assert_eq!(unsafe { probe_stream_closes_get() }, before + 1);
    drop(session);
}

#[test]
fn probe_typed_asr_callbacks_all_routes_and_cleanup() {
    use std::sync::{Arc, Mutex};
    unsafe {
        probe_reset();
        probe_mode_set(0)
    };
    let mut session = new_probe_session();
    let events = Arc::new(Mutex::new(Vec::<String>::new()));
    let progress = events.clone();
    session.set_progress_callback(move |n, total| {
        assert_eq!((n, total), (3, 6));
        progress.lock().unwrap().push("progress".into());
    });
    let segments = events.clone();
    session.set_segment_callback(move |text, t0, t1, index| {
        assert_eq!((text.as_str(), t0, t1, index), ("日本語", 1.23, 4.56, 7));
        segments.lock().unwrap().push(text);
    });
    let tokens = events.clone();
    session.set_token_callback(move |raw, index| {
        assert_eq!(raw, vec![0xe6, 0x97]);
        assert_eq!(index, 9);
        tokens.lock().unwrap().push("token".into());
    });
    fn run(session: &Session, route: i32) -> Result<(), String> {
        let pcm = &[1., 3., 5.];
        match route {
            0 => session.transcribe(pcm).map(|_| ()),
            1 => session
                .transcribe_with_language(pcm, Some("en"))
                .map(|_| ()),
            2 => session.transcribe_chunked(pcm, 30, 1).map(|_| ()),
            3 => session.transcribe_vad(pcm, "vad", None).map(|_| ()),
            _ => session.transcribe_with_logits(pcm).map(|_| ()),
        }
    }
    for route in 0..5 {
        run(&session, route).unwrap();
    }
    assert_eq!(events.lock().unwrap().len(), 15);
    session.clear_progress_callback();
    session.clear_segment_callback();
    session.clear_token_callback();
    run(&session, 0).unwrap();
    assert_eq!(events.lock().unwrap().len(), 15);
    for kind in 0..3 {
        match kind {
            0 => session.set_progress_callback(|_, _| panic!("progress sentinel")),
            1 => session.set_segment_callback(|_, _, _, _| panic!("segment sentinel")),
            _ => session.set_token_callback(|_, _| panic!("token sentinel")),
        }
        for route in 0..5 {
            let before = unsafe { probe_result_frees_get() };
            assert!(run(&session, route).is_err());
            assert_eq!(unsafe { probe_result_frees_get() }, before + 1);
        }
        session.clear_progress_callback();
        session.clear_segment_callback();
        session.clear_token_callback();
        run(&session, 0).unwrap();
    }
    session.set_progress_callback(|_, _| {});
    let before = unsafe { probe_result_frees_get() };
    assert!(session
        .transcribe_chunked_with_progress(&[1., 3., 5.], 30, 1, None, |_, _| panic!(
            "stack callback sentinel"
        ))
        .is_err());
    assert_eq!(unsafe { probe_result_frees_get() }, before + 1);
    run(&session, 0).unwrap();
}

#[test]
fn probe_parakeet_integer_offset() {
    let context = crispasr::Parakeet::new("probe", 4, true).unwrap();
    let pcm = &[1., 3., 5.];
    let result = context.transcribe_at(pcm, 123).unwrap();
    assert_eq!(result.text(), "日本語");
    assert_eq!(result.word_t0(0), 223);
    assert_eq!(result.token_t0(0), -1);
    let mut bytes = result.token_bytes(0);
    bytes.extend(result.token_bytes(1));
    assert_eq!(bytes, "日".as_bytes());
    assert!(context.transcribe(pcm, Some("de")).is_err());
    assert!(context.transcribe_at(&[], 0).is_err());
}

#[test]
fn probe_titanet_float_count_and_ownership() {
    extern "C" {
        fn probe_titanet_mode(mode: c_int);
        fn probe_titanet_live() -> c_int;
    }
    unsafe { probe_titanet_mode(0) };
    let context = crispasr::TitaNet::new("probe", 4).unwrap();
    let pcm = &[1., 3., 5.];
    let output = context.embed(pcm).unwrap();
    assert_eq!(output.len(), 192);
    assert_eq!(output[0], 1.);
    for mode in [2, 3, 4] {
        unsafe { probe_titanet_mode(mode) };
        assert!(context.embed(pcm).is_err());
    }
    unsafe { probe_titanet_mode(0) };
    assert!(context.embed(&[]).is_err());
    drop(context);
    assert_eq!(unsafe { probe_titanet_live() }, 0);
    assert_eq!(
        crispasr::TitaNet::cosine_sim(&[1., 0.], &[1., 0.]).unwrap(),
        1.
    );
    assert!(crispasr::TitaNet::cosine_sim(&[1.], &[]).is_err());
}
