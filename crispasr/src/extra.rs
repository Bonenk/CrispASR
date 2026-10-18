//! Extra session operations — music, conversion, TTS extras, audio helpers.
//! This file is included from lib.rs to keep the main file manageable.
//! All returned native views are copied before the next native call can replace session-owned storage.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};

use crate::Session;

// --- Music types ---

#[derive(Debug, Clone)]
pub struct PianoNote {
    pub onset_ms: f32,
    pub offset_ms: f32,
    pub midi_note: i32,
    pub velocity: i32,
}

#[derive(Debug, Clone)]
pub struct PianoResult {
    pub notes: Vec<PianoNote>,
    pub programs: Vec<i32>,
    pub sample_rate: i32,
}

#[derive(Debug, Clone)]
pub struct ChordSpan {
    pub start_ms: f32,
    pub end_ms: f32,
    pub label: i32,
    pub confidence: f32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct ChordResult {
    pub spans: Vec<ChordSpan>,
    pub vocab_size: i32,
}

#[derive(Debug, Clone)]
pub struct BeatEvent {
    pub time_s: f32,
    pub is_downbeat: bool,
}

#[derive(Debug, Clone)]
pub struct BeatResult {
    pub events: Vec<BeatEvent>,
    pub tempo_bpm: f32,
    pub sample_rate: i32,
}

#[derive(Debug, Clone)]
pub struct TabResult {
    pub emissions: Vec<f32>,
    pub frames: i32,
    pub strings: i32,
    pub classes: i32,
    pub silent_class: i32,
    pub frame_period: f32,
    pub open_midi: Vec<i32>,
}

#[derive(Debug, Clone)]
pub struct PitchFrame {
    pub time_ms: f32,
    pub f0_hz: f32,
    pub voiced_prob: f32,
}

impl Session {
    /// Transcribe mono PCM at piano_sample_rate(); empty notes are valid (not an error).
    pub fn piano(&self, pcm: &[f32]) -> Result<PianoResult, String> {
        if pcm.is_empty() {
            return Err("piano: empty PCM".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_piano(self.handle, pcm.as_ptr(), pcm.len() as c_int)
        };
        if n < 0 {
            return Err(format!("piano failed (rc={})", n));
        }
        if n == 0 {
            return Ok(PianoResult {
                notes: Vec::new(),
                programs: Vec::new(),
                sample_rate: self.piano_sample_rate(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_piano_notes(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("piano: negative note count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("piano: notes pointer is null".to_string());
            }
            // count==0 with null ptr is valid empty result
        }
        let notes = if count == 0 {
            Vec::new()
        } else {
            let len = count
                .checked_mul(4)
                .ok_or("piano: count overflow".to_string())?;
            let flat = unsafe { std::slice::from_raw_parts(ptr, len) };
            let mut v = Vec::with_capacity(count);
            for i in 0..count {
                v.push(PianoNote {
                    onset_ms: flat[i * 4 + 0],
                    offset_ms: flat[i * 4 + 1],
                    midi_note: flat[i * 4 + 2] as i32,
                    velocity: flat[i * 4 + 3] as i32,
                });
            }
            v
        };
        let mut out_n2 = 0;
        let ptr2 =
            unsafe { crispasr_sys::crispasr_session_piano_note_programs(self.handle, &mut out_n2) };
        if out_n2 < 0 {
            return Err(format!("piano: negative programs count {}", out_n2));
        }
        if out_n2 as usize != count {
            return Err(format!(
                "piano notes/programs length mismatch {} vs {}",
                count, out_n2
            ));
        }
        if ptr2.is_null() && count != 0 {
            return Err("piano: programs pointer is null".to_string());
        }
        let progs = if count == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(ptr2, count).to_vec() }
        };
        Ok(PianoResult {
            notes,
            programs: progs,
            sample_rate: self.piano_sample_rate(),
        })
    }

    pub fn piano_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_piano_sample_rate(self.handle) as i32 }
    }

    pub fn piano_n_notes(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_piano_n_notes(self.handle) as i32 }
    }

    pub fn chords(&self, pcm: &[f32], sample_rate: i32) -> Result<ChordResult, String> {
        if pcm.is_empty() {
            return Err("chords: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("chords: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_chords(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!("chords failed (rc={})", n));
        }
        if n == 0 {
            return Ok(ChordResult {
                spans: Vec::new(),
                vocab_size: self.chords_vocab_size(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_chords_spans(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("chords: negative span count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() && count != 0 {
            return Err("chords: spans pointer is null".to_string());
        }
        let flat = if count == 0 {
            Vec::new()
        } else {
            let len = count
                .checked_mul(4)
                .ok_or("chords: count overflow".to_string())?;
            unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
        };
        let mut spans = Vec::with_capacity(count);
        for i in 0..count {
            let name_ptr =
                unsafe { crispasr_sys::crispasr_session_chords_span_name(self.handle, i as c_int) };
            let name = if name_ptr.is_null() {
                String::new()
            } else {
                unsafe { CStr::from_ptr(name_ptr).to_string_lossy().into_owned() }
            };
            spans.push(ChordSpan {
                start_ms: flat[i * 4 + 0],
                end_ms: flat[i * 4 + 1],
                label: flat[i * 4 + 2] as i32,
                confidence: flat[i * 4 + 3],
                name,
            });
        }
        Ok(ChordResult {
            spans,
            vocab_size: self.chords_vocab_size(),
        })
    }

    pub fn chords_vocab_size(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_chords_vocab_size(self.handle) as i32 }
    }

    pub fn beats(&self, pcm: &[f32], sample_rate: i32) -> Result<BeatResult, String> {
        if pcm.is_empty() {
            return Err("beats: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("beats: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_beats(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!(
                "beats failed (rc={}) — sample rate mismatch or no beat arm",
                n
            ));
        }
        if n == 0 {
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_beats_events(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("beats: negative event count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("beats: events pointer is null".to_string());
            }
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        if count == 0 {
            return Ok(BeatResult {
                events: Vec::new(),
                tempo_bpm: self.beats_tempo_bpm(),
                sample_rate: self.beats_sample_rate(),
            });
        }
        let len = count
            .checked_mul(2)
            .ok_or("beats: count overflow".to_string())?;
        let flat = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        let mut events = Vec::with_capacity(count);
        for i in 0..count {
            events.push(BeatEvent {
                time_s: flat[i * 2 + 0],
                is_downbeat: flat[i * 2 + 1] != 0.0,
            });
        }
        Ok(BeatResult {
            events,
            tempo_bpm: self.beats_tempo_bpm(),
            sample_rate: self.beats_sample_rate(),
        })
    }

    pub fn beats_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_beats_sample_rate(self.handle) as i32 }
    }

    pub fn beats_tempo_bpm(&self) -> f32 {
        unsafe { crispasr_sys::crispasr_session_beats_tempo_bpm(self.handle) }
    }

    pub fn tab(&self, pcm: &[f32], sample_rate: i32) -> Result<TabResult, String> {
        if pcm.is_empty() {
            return Err("tab: empty PCM".to_string());
        }
        if sample_rate <= 0 {
            return Err("tab: sample_rate must be positive".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_tab(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                sample_rate,
            )
        };
        if n < 0 {
            return Err(format!("tab failed (rc={})", n));
        }
        if n == 0 {
            return Ok(TabResult {
                emissions: Vec::new(),
                frames: 0,
                strings: 0,
                classes: 0,
                silent_class: self.tab_silent_class(),
                frame_period: self.tab_frame_period(),
                open_midi: Vec::new(),
            });
        }
        let mut out_frames = 0;
        let mut out_strings = 0;
        let mut out_classes = 0;
        let ptr = unsafe {
            crispasr_sys::crispasr_session_tab_emissions(
                self.handle,
                &mut out_frames,
                &mut out_strings,
                &mut out_classes,
            )
        };
        if out_frames < 0 || out_strings < 0 || out_classes < 0 {
            return Err(format!(
                "tab: negative dimensions {}/{}/{}",
                out_frames, out_strings, out_classes
            ));
        }
        if ptr.is_null() {
            return Err("tab: emissions pointer is null".to_string());
        }
        let frames = out_frames as usize;
        let strs = out_strings as usize;
        let cls = out_classes as usize;
        let total = frames
            .checked_mul(strs)
            .and_then(|v| v.checked_mul(cls))
            .ok_or("tab: dimensions overflow".to_string())?;
        if total == 0 {
            // Empty emissions but non-zero dims should have been n==0 early; total==0 means empty
            // but we already handled n==0 above, so this would be unexpected zero product
            // Still safe to return empty vec without calling from_raw_parts
            return Ok(TabResult {
                emissions: Vec::new(),
                frames: out_frames as i32,
                strings: out_strings as i32,
                classes: out_classes as i32,
                silent_class: self.tab_silent_class(),
                frame_period: self.tab_frame_period(),
                open_midi: Vec::new(),
            });
        }
        let emissions = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };
        let mut open_midi = Vec::with_capacity(strs);
        for i in 0..strs {
            open_midi.push(unsafe {
                crispasr_sys::crispasr_session_tab_string_open_midi(self.handle, i as c_int) as i32
            });
        }
        Ok(TabResult {
            emissions,
            frames: out_frames as i32,
            strings: out_strings as i32,
            classes: out_classes as i32,
            silent_class: self.tab_silent_class(),
            frame_period: self.tab_frame_period(),
            open_midi,
        })
    }

    pub fn tab_silent_class(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_tab_silent_class(self.handle) as i32 }
    }

    pub fn tab_frame_period(&self) -> f32 {
        unsafe { crispasr_sys::crispasr_session_tab_frame_period(self.handle) }
    }

    pub fn pitch(&self, pcm: &[f32], hop_ms: f32) -> Result<Vec<PitchFrame>, String> {
        if pcm.is_empty() {
            return Err("pitch: empty PCM".to_string());
        }
        let n = unsafe {
            crispasr_sys::crispasr_session_pitch(
                self.handle,
                pcm.as_ptr(),
                pcm.len() as c_int,
                hop_ms,
            )
        };
        if n < 0 {
            return Err(format!("pitch failed (rc={})", n));
        }
        if n == 0 {
            return Ok(Vec::new());
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_pitch_frames(self.handle, &mut out_n) };
        if out_n < 0 {
            return Err(format!("pitch: negative frame count {}", out_n));
        }
        let count = out_n as usize;
        if ptr.is_null() {
            if count != 0 {
                return Err("pitch: frames pointer is null".to_string());
            }
            return Ok(Vec::new());
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let len = count
            .checked_mul(3)
            .ok_or("pitch: count overflow".to_string())?;
        let flat = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
        let mut frames = Vec::with_capacity(count);
        for i in 0..count {
            frames.push(PitchFrame {
                time_ms: flat[i * 3 + 0],
                f0_hz: flat[i * 3 + 1],
                voiced_prob: flat[i * 3 + 2],
            });
        }
        Ok(frames)
    }

    pub fn pitch_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_pitch_sample_rate(self.handle) as i32 }
    }

    // --- Conversion (RVC) ---

    pub fn convert_content_dim(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_content_dim(self.handle) as i32 }
    }

    pub fn convert_n_speakers(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_n_speakers(self.handle) as i32 }
    }

    pub fn convert_sample_rate(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_sample_rate(self.handle) as i32 }
    }

    pub fn convert_inter_channels(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_inter_channels(self.handle) as i32 }
    }

    pub fn convert_upsample_product(&self) -> i32 {
        unsafe { crispasr_sys::crispasr_session_convert_upsample_product(self.handle) as i32 }
    }

    /// Convert ContentVec features to mono PCM. Validates replay noise lengths using model dimension getters.
    pub fn convert(
        &self,
        content: &[f32],
        n_frames: i32,
        f0_hz: &[f32],
        speaker_id: i32,
        noise_zp: Option<&[f32]>,
        noise_sine: Option<&[f32]>,
    ) -> Result<Vec<f32>, String> {
        if n_frames <= 0 {
            return Err("convert: n_frames must be positive".to_string());
        }
        let dim = self.convert_content_dim();
        if dim <= 0 {
            return Err(
                "convert: could not query content dim (is this a conversion model?)".to_string(),
            );
        }
        let dim_usize = dim as usize;
        let n_frames_usize = n_frames as usize;
        let exp_content = dim_usize
            .checked_mul(n_frames_usize)
            .ok_or("convert: content dimensions overflow".to_string())?;
        if content.len() != exp_content {
            return Err(format!(
                "convert: content length {} != n_frames {} * content_dim {}",
                content.len(),
                n_frames,
                dim
            ));
        }
        if f0_hz.len() != n_frames_usize {
            return Err(format!(
                "convert: f0 length {} != n_frames {}",
                f0_hz.len(),
                n_frames
            ));
        }
        let n_speakers = self.convert_n_speakers();
        if speaker_id < 0 || speaker_id >= n_speakers {
            return Err(format!(
                "convert: speaker_id {} out of range [0,{})",
                speaker_id, n_speakers
            ));
        }
        let inter_ch = self.convert_inter_channels();
        let upsample = self.convert_upsample_product();
        if inter_ch < 0 || upsample < 0 {
            return Err("convert: negative inter_channels or upsample_product".to_string());
        }
        if let Some(zp) = noise_zp {
            let exp_zp = (inter_ch as usize)
                .checked_mul(n_frames_usize)
                .ok_or("convert: noise_zp dimensions overflow".to_string())?;
            if zp.len() != exp_zp {
                return Err(format!(
                    "convert: noise_zp length {} != inter_channels {} * n_frames {}",
                    zp.len(),
                    inter_ch,
                    n_frames
                ));
            }
        }
        if let Some(sine) = noise_sine {
            let exp_sine = n_frames_usize
                .checked_mul(upsample as usize)
                .ok_or("convert: noise_sine dimensions overflow".to_string())?;
            if sine.len() != exp_sine {
                return Err(format!(
                    "convert: noise_sine length {} != n_frames {} * upsample_product {}",
                    sine.len(),
                    n_frames,
                    upsample
                ));
            }
        }
        let zp_ptr = noise_zp.map_or(std::ptr::null(), |v| v.as_ptr());
        let sine_ptr = noise_sine.map_or(std::ptr::null(), |v| v.as_ptr());
        let n_out = unsafe {
            crispasr_sys::crispasr_session_convert(
                self.handle,
                content.as_ptr(),
                n_frames as c_int,
                f0_hz.as_ptr(),
                speaker_id as c_int,
                zp_ptr,
                sine_ptr,
            )
        };
        if n_out <= 0 {
            return Err(format!("convert failed (rc={})", n_out));
        }
        let mut out_n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_convert_audio(self.handle, &mut out_n) };
        if out_n <= 0 {
            return Err(format!("convert: invalid audio length {}", out_n));
        }
        if ptr.is_null() {
            return Err("convert: audio pointer is null".to_string());
        }
        let audio = unsafe { std::slice::from_raw_parts(ptr, out_n as usize).to_vec() };
        Ok(audio)
    }

    // --- TTS extras ---

    pub fn set_tts_cfg_scale(&self, scale: f32) -> Result<(), String> {
        let rc = unsafe { crispasr_sys::crispasr_session_set_tts_cfg_scale(self.handle, scale) };
        if rc == -2 {
            return Err(format!(
                "set_tts_cfg_scale not supported by backend (rc={})",
                rc
            ));
        }
        if rc != 0 {
            return Err(format!("set_tts_cfg_scale failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn tada_set_makeref_models(
        &self,
        encoder: Option<&str>,
        aligner: Option<&str>,
    ) -> Result<(), String> {
        let enc_c = encoder.map(|s| CString::new(s).unwrap());
        let ali_c = aligner.map(|s| CString::new(s).unwrap());
        let enc_ptr = enc_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let ali_ptr = ali_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
        let rc = unsafe {
            crispasr_sys::crispasr_session_tada_set_makeref_models(self.handle, enc_ptr, ali_ptr)
        };
        if rc != 0 {
            return Err(format!("tada_set_makeref_models failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn disclaimer_text() -> String {
        let ptr = unsafe { crispasr_sys::crispasr_session_disclaimer_text() };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    pub fn get_disclaimer_pcm(&self) -> Result<Vec<f32>, String> {
        let mut n = 0;
        let ptr = unsafe { crispasr_sys::crispasr_session_get_disclaimer_pcm(self.handle, &mut n) };
        if n <= 0 {
            // Ensure no leak if native returned allocated buffer with n<=0
            if !ptr.is_null() {
                unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
            }
            return Err("get_disclaimer_pcm: no audio (call before SetVoice?)".to_string());
        }
        if ptr.is_null() {
            return Err("get_disclaimer_pcm: null pointer with positive length".to_string());
        }
        let audio =
            unsafe { std::slice::from_raw_parts(ptr as *const c_float, n as usize).to_vec() };
        unsafe { crispasr_sys::crispasr_pcm_free(ptr) };
        Ok(audio)
    }

    pub fn last_synth_error(&self) -> String {
        let ptr = unsafe { crispasr_sys::crispasr_session_last_synth_error(self.handle) };
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
    }

    /// Synchronous sentence streaming TTS. Fires cb(pcm, is_final) per sentence chunk.
    /// PCM is copied before callback returns; panics are caught and returned as errors.
    pub fn synthesize_streaming<F>(&mut self, text: &str, mut cb: F) -> Result<(), String>
    where
        F: FnMut(&[f32], bool),
    {
        let ctext = CString::new(text).map_err(|e| e.to_string())?;
        // Wrap callback to handle panic and copy
        struct Ctx<F> {
            cb: F,
            err: Option<String>,
        }
        unsafe extern "C" fn trampoline<F: FnMut(&[f32], bool)>(
            pcm: *const c_float,
            n: c_int,
            is_final: c_int,
            user_data: *mut std::os::raw::c_void,
        ) {
            if user_data.is_null() {
                return;
            }
            let ctx = &mut *(user_data as *mut Ctx<F>);
            if ctx.err.is_some() {
                return;
            }
            if pcm.is_null() || n <= 0 {
                ctx.err = Some("synthesize_streaming: empty chunk from native".to_string());
                return;
            }
            // Copy PCM before callback returns
            let src = std::slice::from_raw_parts(pcm, n as usize);
            let cp = src.to_vec();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                (ctx.cb)(&cp, is_final != 0);
            }));
            if result.is_err() {
                ctx.err = Some("synthesize_streaming callback panicked".to_string());
            }
        }
        let mut ctx = Ctx { cb, err: None };
        let rc = unsafe {
            crispasr_sys::crispasr_session_synthesize_streaming(
                self.handle,
                ctext.as_ptr(),
                Some(trampoline::<F>),
                &mut ctx as *mut _ as *mut std::os::raw::c_void,
            )
        };
        if let Some(e) = ctx.err {
            return Err(e);
        }
        if rc != 0 {
            return Err(format!("synthesize_streaming failed (rc={})", rc));
        }
        Ok(())
    }

    // --- Audio helpers (standalone, not session-bound) ---

    // Session-bound audio rate helpers already exist as input_sample_rate etc.

    pub fn set_pcm_sample_rate(&self, rate: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_pcm_sample_rate(self.handle, rate as c_int)
        };
        if rc != 0 {
            return Err(format!("set_pcm_sample_rate failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_parakeet_att_context(&self, left: i32, right: i32) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_parakeet_att_context(
                self.handle,
                left as c_int,
                right as c_int,
            )
        };
        if rc != 0 {
            return Err(format!("set_parakeet_att_context failed (rc={})", rc));
        }
        Ok(())
    }

    pub fn set_grammar_strict(&self, strict: bool) -> Result<(), String> {
        let rc = unsafe {
            crispasr_sys::crispasr_session_set_grammar_strict(
                self.handle,
                if strict { 1 } else { 0 },
            )
        };
        if rc != 0 {
            return Err(format!("set_grammar_strict failed (rc={})", rc));
        }
        Ok(())
    }
}

// Standalone functions

// Native audio buffers are owned immediately, including partial error outputs.
struct OwnedNativeAudio(*mut c_float);
impl Drop for OwnedNativeAudio {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { crispasr_sys::crispasr_audio_free(self.0) }
        }
    }
}
fn copy_native_audio(p: *const c_float, n: c_int) -> Result<Vec<f32>, String> {
    if n < 0 || (n > 0 && p.is_null()) {
        return Err("invalid native PCM buffer".into());
    }
    if n == 0 {
        return Ok(Vec::new());
    }
    Ok(unsafe { std::slice::from_raw_parts(p, n as usize).to_vec() })
}
pub fn audio_load(path: &str, target_rate: Option<i32>) -> Result<(Vec<f32>, i32), String> {
    let path = CString::new(path).map_err(|e| e.to_string())?;
    if target_rate.is_some_and(|r| r <= 0) {
        return Err("positive target rate required".into());
    }
    let (mut p, mut n, mut rate) = (std::ptr::null_mut(), 0, 0);
    let rc = unsafe {
        if let Some(target) = target_rate {
            crispasr_sys::crispasr_audio_load_at_rate(
                path.as_ptr(),
                target,
                &mut p,
                &mut n,
                &mut rate,
            )
        } else {
            crispasr_sys::crispasr_audio_load(path.as_ptr(), &mut p, &mut n, &mut rate)
        }
    };
    let _owned = OwnedNativeAudio(p);
    if rc != 0 {
        return Err(format!("audio_load failed (rc={rc})"));
    }
    Ok((copy_native_audio(p, n)?, rate))
}
pub fn audio_load_stereo(path: &str) -> Result<(Vec<f32>, Vec<f32>, i32, i32), String> {
    let path = CString::new(path).map_err(|e| e.to_string())?;
    let (mut l, mut r, mut n, mut rate, mut channels) =
        (std::ptr::null_mut(), std::ptr::null_mut(), 0, 0, 0);
    let rc = unsafe {
        crispasr_sys::crispasr_audio_load_stereo(
            path.as_ptr(),
            &mut l,
            &mut r,
            &mut n,
            &mut rate,
            &mut channels,
        )
    };
    let _left = OwnedNativeAudio(l);
    let _right = OwnedNativeAudio(if l == r { std::ptr::null_mut() } else { r });
    if rc != 0 {
        return Err(format!("audio_load_stereo failed (rc={rc})"));
    }
    Ok((
        copy_native_audio(l, n)?,
        copy_native_audio(r, n)?,
        rate,
        channels,
    ))
}
pub fn audio_resample(pcm: &[f32], source_rate: i32, target_rate: i32) -> Result<Vec<f32>, String> {
    if source_rate <= 0 || target_rate <= 0 {
        return Err("positive sample rates required".into());
    }
    if pcm.is_empty() {
        return Ok(Vec::new());
    }
    let count = c_int::try_from(pcm.len()).map_err(|_| "PCM exceeds native count range")?;
    let (mut p, mut n) = (std::ptr::null_mut(), 0);
    let rc = unsafe {
        crispasr_sys::crispasr_audio_resample(
            pcm.as_ptr(),
            count,
            source_rate,
            target_rate,
            &mut p,
            &mut n,
        )
    };
    let _owned = OwnedNativeAudio(p);
    if rc != 0 {
        return Err(format!("audio_resample failed (rc={rc})"));
    }
    copy_native_audio(p, n)
}

pub fn pcm_to_wav(pcm: &[f32], sample_rate: i32) -> Result<Vec<u8>, String> {
    if pcm.is_empty() {
        return Err("pcm_to_wav: empty PCM".to_string());
    }
    if sample_rate <= 0 {
        return Err("pcm_to_wav: invalid sample_rate".to_string());
    }
    let mut out_len: usize = 0;
    let ptr = unsafe {
        crispasr_sys::crispasr_pcm_to_wav(
            pcm.as_ptr(),
            pcm.len() as c_int,
            sample_rate as c_int,
            &mut out_len,
        )
    };
    if out_len == 0 {
        if !ptr.is_null() {
            unsafe { crispasr_sys::crispasr_c2pa_free(ptr) };
        }
        return Err("pcm_to_wav failed".to_string());
    }
    if ptr.is_null() {
        return Err("pcm_to_wav: null pointer with positive length".to_string());
    }
    let wav = unsafe { std::slice::from_raw_parts(ptr, out_len).to_vec() };
    unsafe { crispasr_sys::crispasr_c2pa_free(ptr) };
    Ok(wav)
}

pub fn c2pa_sign(
    data: &[u8],
    format: &str,
    cert_path: Option<&str>,
    key_path: Option<&str>,
) -> Result<Vec<u8>, String> {
    if data.is_empty() {
        return Err("c2pa_sign: empty data".to_string());
    }
    let cformat = CString::new(format).map_err(|e| e.to_string())?;
    let ccert = cert_path.map(|s| CString::new(s).unwrap());
    let ckey = key_path.map(|s| CString::new(s).unwrap());
    let cert_ptr = ccert.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let key_ptr = ckey.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    let mut out_len: usize = 0;
    let ptr = unsafe {
        crispasr_sys::crispasr_c2pa_sign(
            data.as_ptr(),
            data.len(),
            cformat.as_ptr(),
            cert_ptr,
            key_ptr,
            &mut out_len,
        )
    };
    if out_len == 0 {
        if !ptr.is_null() {
            unsafe { crispasr_sys::crispasr_c2pa_free(ptr) };
        }
        return Err("c2pa_sign failed (unavailable or bad container)".to_string());
    }
    if ptr.is_null() {
        return Err("c2pa_sign: null pointer with positive length".to_string());
    }
    let signed = unsafe { std::slice::from_raw_parts(ptr, out_len).to_vec() };
    unsafe { crispasr_sys::crispasr_c2pa_free(ptr) };
    Ok(signed)
}

pub fn watermark_load_model(gguf_path: &str) -> Result<(), String> {
    let cpath = CString::new(gguf_path).map_err(|e| e.to_string())?;
    let rc = unsafe { crispasr_sys::crispasr_watermark_load_model(cpath.as_ptr()) };
    if rc != 0 {
        return Err(format!("watermark_load_model failed (rc={})", rc));
    }
    Ok(())
}

pub fn free_aligner_cache() {
    unsafe { crispasr_sys::crispasr_free_aligner_cache() }
}

pub fn set_gpu_backend(name: Option<&str>) {
    let cname = name.map(|s| CString::new(s).unwrap());
    let ptr = cname.as_ref().map_or(std::ptr::null(), |c| c.as_ptr());
    unsafe { crispasr_sys::crispasr_set_gpu_backend(ptr) }
}

pub fn detect_backends_from_gguf(path: &str) -> Result<Vec<String>, String> {
    let cpath = CString::new(path).map_err(|e| e.to_string())?;
    let mut buf = vec![0i8; 512];
    let n = unsafe {
        crispasr_sys::crispasr_detect_backends_from_gguf(
            cpath.as_ptr(),
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        )
    };
    if n <= 0 {
        return Err(format!("detect_backends_from_gguf failed (rc={})", n));
    }
    let s = unsafe {
        CStr::from_ptr(buf.as_ptr() as *const c_char)
            .to_string_lossy()
            .into_owned()
    };
    Ok(s.lines()
        .filter(|l| !l.is_empty())
        .map(|l| l.to_string())
        .collect())
}

pub fn c_api_version() -> String {
    let ptr = unsafe { crispasr_sys::crispasr_c_api_version() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}

pub fn dart_helpers_version() -> String {
    let ptr = unsafe { crispasr_sys::crispasr_dart_helpers_version() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr).to_string_lossy().into_owned() }
}
