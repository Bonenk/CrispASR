//! Per-call controls for the deprecated Whisper context; strings remain owned
//! until native inference and parameter destruction complete.
use std::ffi::CString;

/// Optional legacy Whisper scalar controls. `None` preserves the native default;
/// `Some(false)` explicitly clears a flag. Construct with `Default::default()`
/// and assign the public fields; the struct is non-exhaustive for future controls.
/// Existing `TranscribeOptions` still carries strategy, VAD and tinydiarize.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct WhisperDecodeControls {
    /// Native language code/name or "auto". Must be nonempty and NUL-free.
    pub language: Option<String>,
    /// NUL-free prompt; an empty string explicitly clears the prompt.
    pub initial_prompt: Option<String>,
    pub translate: Option<bool>,
    pub detect_language: Option<bool>,
    pub token_timestamps: Option<bool>,
    pub split_on_word: Option<bool>,
    pub no_context: Option<bool>,
    pub single_segment: Option<bool>,
    pub print_realtime: Option<bool>,
    pub print_progress: Option<bool>,
    pub print_timestamps: Option<bool>,
    pub print_special: Option<bool>,
    pub suppress_blank: Option<bool>,

    /// Positive worker count.
    pub n_threads: Option<i32>,
    /// Maximum segment length; zero leaves length unlimited. Must be nonnegative.
    pub max_len: Option<i32>,
    /// Positive greedy candidate count.
    pub best_of: Option<i32>,
    /// Finite, nonnegative decoding temperature.
    pub temperature: Option<f32>,
    /// Maximum tokens per segment; zero means unlimited. Must be nonnegative.
    pub max_tokens: Option<i32>,
    /// Greedy token alternatives; native setter clamps to 0..32. Beam decoding
    /// does not emit alternatives even when this field is set.
    pub alt_n: Option<i32>,
}

pub(crate) struct PreparedWhisperControls<'a> {
    controls: &'a WhisperDecodeControls,
    language: Option<CString>,
    prompt: Option<CString>,
}

impl WhisperDecodeControls {
    pub(crate) fn prepare(&self) -> Result<PreparedWhisperControls<'_>, String> {
        if self.n_threads.is_some_and(|n| n <= 0) || self.best_of.is_some_and(|n| n <= 0) {
            return Err("threads and best-of must be positive".into());
        }
        if self.max_len.is_some_and(|n| n < 0) || self.max_tokens.is_some_and(|n| n < 0) {
            return Err("length and token limits must be nonnegative".into());
        }
        if self.temperature.is_some_and(|t| !t.is_finite() || t < 0.) {
            return Err("temperature must be finite and nonnegative".into());
        }
        if self.language.as_ref().is_some_and(|s| s.is_empty()) {
            return Err("language must be nonempty".into());
        }
        let language = self
            .language
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()
            .map_err(|_| "language contains NUL")?;
        let prompt = self
            .initial_prompt
            .as_ref()
            .map(|s| CString::new(s.as_str()))
            .transpose()
            .map_err(|_| "prompt contains NUL")?;
        Ok(PreparedWhisperControls {
            controls: self,
            language,
            prompt,
        })
    }
}

impl PreparedWhisperControls<'_> {
    /// Caller owns a live, uniquely configured native full-params allocation.
    /// Keep this owner alive until native inference and matching free finish.
    pub(crate) unsafe fn apply(&self, params: *mut crispasr_sys::WhisperFullParams) {
        macro_rules! flag {
            ($field:ident, $setter:ident) => {
                if let Some(value) = self.controls.$field {
                    unsafe { crispasr_sys::$setter(params, i32::from(value)) };
                }
            };
        }
        macro_rules! scalar {
            ($field:ident, $setter:ident) => {
                if let Some(value) = self.controls.$field {
                    unsafe { crispasr_sys::$setter(params, value) };
                }
            };
        }
        flag!(translate, crispasr_params_set_translate);
        flag!(detect_language, crispasr_params_set_detect_language);
        flag!(token_timestamps, crispasr_params_set_token_timestamps);
        flag!(split_on_word, crispasr_params_set_split_on_word);
        flag!(no_context, crispasr_params_set_no_context);
        flag!(single_segment, crispasr_params_set_single_segment);
        flag!(print_realtime, crispasr_params_set_print_realtime);
        flag!(print_progress, crispasr_params_set_print_progress);
        flag!(print_timestamps, crispasr_params_set_print_timestamps);
        flag!(print_special, crispasr_params_set_print_special);
        flag!(suppress_blank, crispasr_params_set_suppress_blank);
        scalar!(n_threads, crispasr_params_set_n_threads);
        scalar!(max_len, crispasr_params_set_max_len);
        scalar!(best_of, crispasr_params_set_best_of);
        scalar!(temperature, crispasr_params_set_temperature);
        scalar!(max_tokens, crispasr_params_set_max_tokens);
        scalar!(alt_n, crispasr_params_set_alt_n);

        if let Some(language) = &self.language {
            unsafe { crispasr_sys::crispasr_params_set_language(params, language.as_ptr()) };
        }
        if let Some(prompt) = &self.prompt {
            unsafe { crispasr_sys::crispasr_params_set_initial_prompt(params, prompt.as_ptr()) };
        }
    }
}
