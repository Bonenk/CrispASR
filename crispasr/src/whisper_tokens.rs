//! Owned results from the legacy Whisper context. No borrowed pointers escape.
#[allow(deprecated)]
use crate::CrispASR;

const MAX_TOKENS: i32 = 1_000_000;
const MAX_TEXT: usize = 1 << 20;

/// A copied legacy Whisper token alternative. Text can be a UTF-8 fragment.
#[derive(Debug, Clone)]
pub struct WhisperTokenAlternative {
    pub id: i32,
    /// Replacement-decoded display text; use `raw` for lossless reconstruction.
    pub text: String,
    pub raw: Vec<u8>,
    /// Native probability, preserving -1 if unavailable.
    pub probability: f32,
}

/// A copied token from a legacy Whisper context. All times are native
/// centiseconds, preserving -1 when unavailable. Results survive later decode
/// and model destruction; split UTF-8 is preserved in `raw`.
#[derive(Debug, Clone)]
pub struct WhisperToken {
    pub id: i32,
    pub text: String,
    pub raw: Vec<u8>,
    pub t0: i64,
    pub t1: i64,
    pub dtw: i64,
    pub probability: f32,
    pub alternatives: Vec<WhisperTokenAlternative>,
}

#[allow(deprecated)]
impl CrispASR {
    fn legacy_token_count(&self, segment: i32) -> Result<i32, String> {
        if segment < 0 {
            return Err("nonnegative segment index required".into());
        }
        let segments = unsafe { crispasr_sys::whisper_full_n_segments(self.ctx) };
        if segments < 0 || segment >= segments {
            return Err("segment index outside native results".into());
        }
        let count = unsafe { crispasr_sys::whisper_full_n_tokens(self.ctx, segment) };
        if !(0..=MAX_TOKENS).contains(&count) {
            return Err("invalid or excessive native token count".into());
        }
        Ok(count)
    }

    /// Copy one token, including raw bytes and alternatives. Indices and native
    /// counts are checked before indexed access. Tokens/alternatives cap text at
    /// 1 MiB each, and native alternative counts must be between zero and 32.
    pub fn token(&self, segment: i32, token: i32) -> Result<WhisperToken, String> {
        if token < 0 {
            return Err("nonnegative token index required".into());
        }
        if token >= self.legacy_token_count(segment)? {
            return Err("token index outside native results".into());
        }
        self.copy_legacy_token(segment, token)
    }

    /// Copy all tokens in a segment (at most one million). Empty token lists
    /// are valid; invalid segments/counts and malformed alternatives return errors.
    pub fn tokens(&self, segment: i32) -> Result<Vec<WhisperToken>, String> {
        let count = self.legacy_token_count(segment)?;
        let mut result = Vec::new();
        result
            .try_reserve_exact(count as usize)
            .map_err(|_| "could not reserve owned token results")?;
        for token in 0..count {
            result.push(self.copy_legacy_token(segment, token)?);
        }
        Ok(result)
    }

    // Native results belong to this context, which is not Sync. No callbacks or
    // inference run while copying, and native C-string lifetimes cover the copy.
    fn copy_legacy_token(&self, segment: i32, token: i32) -> Result<WhisperToken, String> {
        let ptr = unsafe { crispasr_sys::whisper_full_get_token_text(self.ctx, segment, token) };
        let raw = if ptr.is_null() {
            Vec::new()
        } else {
            // The C ABI promises a valid NUL-terminated borrowed string. Bound
            // the scan as well as the copy for oversized vocabulary entries.
            let mut length = 0;
            while length <= MAX_TEXT && unsafe { *ptr.add(length) } != 0 {
                length += 1;
            }
            if length > MAX_TEXT {
                return Err("native token text exceeds 1 MiB".into());
            }
            unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), length) }.to_vec()
        };
        let count = unsafe { crispasr_sys::crispasr_token_n_alts(self.ctx, segment, token) };
        if !(0..=32).contains(&count) {
            return Err("invalid native alternative count".into());
        }
        let mut alternatives = Vec::with_capacity(count as usize);
        for alternative in 0..count {
            let bytes = self.copy_legacy_alternative_text(segment, token, alternative)?;
            alternatives.push(WhisperTokenAlternative {
                id: unsafe {
                    crispasr_sys::crispasr_token_alt_id(self.ctx, segment, token, alternative)
                },
                probability: unsafe {
                    crispasr_sys::crispasr_token_alt_p(self.ctx, segment, token, alternative)
                },
                text: String::from_utf8_lossy(&bytes).into_owned(),
                raw: bytes,
            });
        }
        Ok(WhisperToken {
            id: unsafe { crispasr_sys::whisper_full_get_token_id(self.ctx, segment, token) },
            text: String::from_utf8_lossy(&raw).into_owned(),
            raw,
            t0: unsafe { crispasr_sys::crispasr_token_t0(self.ctx, segment, token) },
            t1: unsafe { crispasr_sys::crispasr_token_t1(self.ctx, segment, token) },
            dtw: unsafe { crispasr_sys::crispasr_token_dtw_t(self.ctx, segment, token) },
            probability: unsafe { crispasr_sys::crispasr_token_p(self.ctx, segment, token) },
            alternatives,
        })
    }

    fn copy_legacy_alternative_text(
        &self,
        segment: i32,
        token: i32,
        alternative: i32,
    ) -> Result<Vec<u8>, String> {
        let mut capacity = 256;
        loop {
            // Poison the terminator too: a native success must actually write it.
            let mut buffer = vec![0xff; capacity];
            let count = unsafe {
                crispasr_sys::crispasr_token_alt_text(
                    self.ctx,
                    segment,
                    token,
                    alternative,
                    buffer.as_mut_ptr().cast(),
                    capacity as i32,
                )
            };
            if count == -1 && capacity < MAX_TEXT + 1 {
                capacity = (capacity * 2).min(MAX_TEXT + 1);
                continue;
            }
            if count < 0 || count as usize >= capacity {
                return Err("invalid or excessive native alternative text length".into());
            }
            let count = count as usize;
            if buffer[count] != 0 || buffer[..count].contains(&0) {
                return Err("malformed native alternative text".into());
            }
            buffer.truncate(count);
            return Ok(buffer);
        }
    }
}
