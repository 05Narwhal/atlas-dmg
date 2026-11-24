// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! xz streams of LZMA2 data, for UDIF LZMA chunks.
//!
//! [`xz_decode`] decodes one or more concatenated xz streams (with stream
//! padding between them) into a slice of exactly the decoded size. The
//! slice is also the LZMA dictionary: matches copy from earlier bytes of
//! it, so no window is allocated and nothing grows with the input. The
//! only allocation is the probability model, about 28 KiB, made once per
//! call. Writing past the end of the slice, or finishing short of it, is
//! an error.
//!
//! Verified:
//! - stream header and footer: magic, flags, CRC32, backward size, and
//!   footer flags equal to header flags; stream padding in multiples of 4;
//! - block headers: CRC32, reserved bits, exactly one filter which is
//!   LZMA2 (0x21) with a one-byte dictionary property, zero padding, and
//!   the compressed and uncompressed sizes when present;
//! - LZMA2: the first chunk resets the dictionary, properties come before
//!   the first LZMA chunk that needs them (lc + lp <= 4, pb <= 4), every
//!   LZMA chunk decodes exactly its unpacked size from exactly its packed
//!   size and leaves the range coder finished, no match crosses a chunk
//!   end, and no distance reaches before the last dictionary reset or past
//!   the declared dictionary size (which also rules out end markers);
//! - block padding, CRC32 and CRC64 block checks, and the index: record
//!   count, every record, padding and CRC32.
//!
//! Not verified: SHA-256 block checks are skipped (their 32 bytes are
//! consumed). Check types other than none, CRC32, CRC64 and SHA-256 are
//! rejected. Filters other than a lone LZMA2 (BCJ, delta) are rejected.

use super::{LONGER, exact, len64};
use atlas_fs_core::error::CorruptKind;

type Res<T> = Result<T, CorruptKind>;

const MAGIC: &[u8; 6] = b"\xFD7zXZ\x00";
const FOOTER_MAGIC: &[u8; 2] = b"YZ";
const TRUNCATED: CorruptKind = CorruptKind::Other("xz data is truncated");
const DIST: CorruptKind = CorruptKind::Other("LZMA distance reaches before the dictionary start");

const fn bad(text: &'static str) -> CorruptKind {
    CorruptKind::Other(text)
}

// ---- checksums -------------------------------------------------------------

/// Slicing-by-8 tables: `t[k][n]` is the CRC of byte `n` followed by `k`
/// zero bytes.
// Evaluated at compile time: an out-of-range index or an overflow fails the
// build instead of panicking at run time.
#[allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation
)]
const fn crc32_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xEDB8_8320
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[s - 1][i];
            t[s][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            i += 1;
        }
        s += 1;
    }
    t
}

/// As [`crc32_tables`], for CRC64 (ECMA-182, reflected, as xz uses).
#[allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation
)]
const fn crc64_tables() -> [[u64; 256]; 8] {
    let mut t = [[0u64; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u64;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0xC96C_5795_D787_0F42
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[s - 1][i];
            t[s][i] = (prev >> 8) ^ t[0][(prev & 0xFF) as usize];
            i += 1;
        }
        s += 1;
    }
    t
}

static CRC32_TABLES: [[u32; 256]; 8] = crc32_tables();
static CRC64_TABLES: [[u64; 256]; 8] = crc64_tables();

fn look32(t: &[u32; 256], i: u8) -> u32 {
    t.get(usize::from(i)).copied().unwrap_or(0)
}

fn look64(t: &[u64; 256], i: u8) -> u64 {
    t.get(usize::from(i)).copied().unwrap_or(0)
}

/// A running CRC32 (IEEE, as xz and zlib use).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Crc32(u32);

impl Crc32 {
    pub(super) fn new() -> Self {
        Self(!0)
    }

    pub(super) fn of(data: &[u8]) -> u32 {
        let mut c = Self::new();
        c.update(data);
        c.finish()
    }

    pub(super) fn update(&mut self, data: &[u8]) {
        let t = &CRC32_TABLES;
        let mut c = self.0;
        let (chunks, rest) = data.as_chunks::<8>();
        for &[a0, a1, a2, a3, b0, b1, b2, b3] in chunks {
            let [l0, l1, l2, l3] = (u32::from_le_bytes([a0, a1, a2, a3]) ^ c).to_le_bytes();
            c = look32(&t[7], l0)
                ^ look32(&t[6], l1)
                ^ look32(&t[5], l2)
                ^ look32(&t[4], l3)
                ^ look32(&t[3], b0)
                ^ look32(&t[2], b1)
                ^ look32(&t[1], b2)
                ^ look32(&t[0], b3);
        }
        for &b in rest {
            let [low, ..] = c.to_le_bytes();
            c = (c >> 8) ^ look32(&t[0], low ^ b);
        }
        self.0 = c;
    }

    pub(super) fn finish(self) -> u32 {
        !self.0
    }
}

/// CRC64 of `data`.
pub(super) fn crc64(data: &[u8]) -> u64 {
    let t = &CRC64_TABLES;
    let mut c = !0u64;
    let (chunks, rest) = data.as_chunks::<8>();
    for chunk in chunks {
        let [v0, v1, v2, v3, v4, v5, v6, v7] = (u64::from_le_bytes(*chunk) ^ c).to_le_bytes();
        c = look64(&t[7], v0)
            ^ look64(&t[6], v1)
            ^ look64(&t[5], v2)
            ^ look64(&t[4], v3)
            ^ look64(&t[3], v4)
            ^ look64(&t[2], v5)
            ^ look64(&t[1], v6)
            ^ look64(&t[0], v7);
    }
    for &b in rest {
        let [low, ..] = c.to_le_bytes();
        c = (c >> 8) ^ look64(&t[0], low ^ b);
    }
    !c
}

// ---- byte helpers ----------------------------------------------------------

fn le32(b: &[u8], at: usize) -> Res<u32> {
    let bytes = b
        .get(at..at.checked_add(4).ok_or(TRUNCATED)?)
        .ok_or(TRUNCATED)?;
    Ok(u32::from_le_bytes(bytes.try_into().map_err(|_| TRUNCATED)?))
}

fn be16(b: &[u8], at: usize) -> Res<usize> {
    let bytes = b
        .get(at..at.checked_add(2).ok_or(TRUNCATED)?)
        .ok_or(TRUNCATED)?;
    let pair: [u8; 2] = bytes.try_into().map_err(|_| TRUNCATED)?;
    Ok(usize::from(u16::from_be_bytes(pair)))
}

/// `len` bytes at `at`, or truncated.
fn take(b: &[u8], at: usize, len: usize) -> Res<&[u8]> {
    b.get(at..at.checked_add(len).ok_or(TRUNCATED)?)
        .ok_or(TRUNCATED)
}

/// Zero to three bytes that pad `n` to a multiple of four.
fn pad4(n: usize) -> usize {
    n.wrapping_neg() & 3
}

/// An xz variable-length integer: up to nine bytes of seven bits each,
/// little-endian, in the shortest form.
fn varint(b: &[u8], pos: &mut usize) -> Res<u64> {
    let mut value = 0u64;
    for i in 0..9u32 {
        let byte = *b.get(*pos).ok_or(TRUNCATED)?;
        *pos = pos.saturating_add(1);
        value |= u64::from(byte & 0x7F).wrapping_shl(i.wrapping_mul(7));
        if byte & 0x80 == 0 {
            if i > 0 && byte == 0 {
                return Err(bad("xz integer is not in its shortest form"));
            }
            return Ok(value);
        }
    }
    Err(bad("xz integer is longer than nine bytes"))
}

// ---- range decoder ---------------------------------------------------------

const TOP: u32 = 1 << 24;
const PROB_INIT: u16 = 1024;

/// The range decoder of one LZMA chunk, over exactly its packed bytes.
/// Reading past them sets `bad` and yields zeros, so decoding stays bounded
/// by the unpacked size and the chunk is rejected at its end.
struct Rc<'a> {
    input: &'a [u8],
    pos: usize,
    range: u32,
    code: u32,
    bad: bool,
}

impl<'a> Rc<'a> {
    fn new(input: &'a [u8]) -> Res<Self> {
        let init: [u8; 5] = input
            .get(..5)
            .and_then(|b| b.try_into().ok())
            .ok_or(bad("LZMA chunk is shorter than the range coder start"))?;
        let [first, c0, c1, c2, c3] = init;
        if first != 0 {
            return Err(bad("LZMA range coder does not start with zero"));
        }
        let code = u32::from_be_bytes([c0, c1, c2, c3]);
        if code == u32::MAX {
            return Err(bad("LZMA range coder start is invalid"));
        }
        Ok(Self {
            input,
            pos: 5,
            range: u32::MAX,
            code,
            bad: false,
        })
    }

    #[inline(always)]
    fn normalize(&mut self) {
        if self.range < TOP {
            self.range <<= 8;
            let byte = match self.input.get(self.pos) {
                Some(&b) => {
                    self.pos = self.pos.wrapping_add(1);
                    b
                }
                None => {
                    self.bad = true;
                    0
                }
            };
            self.code = (self.code << 8) | u32::from(byte);
        }
    }

    #[inline(always)]
    fn bit(&mut self, p: &mut u16) -> usize {
        let bound = (self.range >> 11).wrapping_mul(u32::from(*p));
        let bit = if self.code < bound {
            self.range = bound;
            *p = p.wrapping_add(2048u16.wrapping_sub(*p) >> 5);
            0
        } else {
            self.code = self.code.wrapping_sub(bound);
            self.range = self.range.wrapping_sub(bound);
            *p = p.wrapping_sub(*p >> 5);
            1
        };
        self.normalize();
        bit
    }

    #[inline(always)]
    fn bit_at(&mut self, probs: &mut [u16], i: usize) -> usize {
        match probs.get_mut(i) {
            Some(p) => self.bit(p),
            None => {
                self.bad = true;
                0
            }
        }
    }

    /// A `bits`-bit symbol, most significant bit first.
    #[inline(always)]
    fn tree(&mut self, probs: &mut [u16], bits: u32) -> usize {
        let mut m = 1usize;
        for _ in 0..bits {
            m = (m << 1) | self.bit_at(probs, m);
        }
        m ^ 1usize.wrapping_shl(bits)
    }

    /// A `bits`-bit symbol, least significant bit first, with its nodes
    /// at `base + 1 ..` in `probs`.
    #[inline(always)]
    fn reverse(&mut self, probs: &mut [u16], base: usize, bits: u32) -> usize {
        let mut m = 1usize;
        let mut symbol = 0usize;
        for i in 0..bits {
            let bit = self.bit_at(probs, base.wrapping_add(m));
            m = (m << 1) | bit;
            symbol |= bit.wrapping_shl(i);
        }
        symbol
    }

    /// `bits` bits of fixed probability one half.
    fn direct(&mut self, bits: u32) -> u32 {
        let mut result = 0u32;
        for _ in 0..bits {
            self.range >>= 1;
            self.code = self.code.wrapping_sub(self.range);
            let t = 0u32.wrapping_sub(self.code >> 31);
            self.code = self.code.wrapping_add(self.range & t);
            if self.code == self.range {
                self.bad = true;
            }
            self.normalize();
            result = (result << 1).wrapping_add(t.wrapping_add(1));
        }
        result
    }

    /// True when every packed byte was used and the coder ended cleanly.
    fn finished(&self) -> bool {
        !self.bad && self.code == 0 && self.pos == self.input.len()
    }
}

// ---- LZMA ------------------------------------------------------------------

const STATES: usize = 12;
const LITERAL_CODER: usize = 0x300;
/// Literal coders for the largest lc + lp LZMA2 allows (4).
const LITERAL_MAX: usize = LITERAL_CODER << 4;
const END_POS_MODEL: usize = 14;
const FULL_DISTANCES: usize = 128;
const ALIGN_BITS: u32 = 4;
const MATCH_MIN: usize = 2;
/// The state after a literal, by state.
const LITERAL_NEXT: [usize; STATES] = [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 4, 5];

struct LenModel {
    choice: u16,
    choice2: u16,
    low: [[u16; 8]; 16],
    mid: [[u16; 8]; 16],
    high: [u16; 256],
}

impl LenModel {
    const fn new() -> Self {
        Self {
            choice: PROB_INIT,
            choice2: PROB_INIT,
            low: [[PROB_INIT; 8]; 16],
            mid: [[PROB_INIT; 8]; 16],
            high: [PROB_INIT; 256],
        }
    }

    /// Match length minus [`MATCH_MIN`].
    fn decode(&mut self, rc: &mut Rc<'_>, pos_state: usize) -> usize {
        if rc.bit(&mut self.choice) == 0 {
            return match self.low.get_mut(pos_state) {
                Some(p) => rc.tree(p, 3),
                None => {
                    rc.bad = true;
                    0
                }
            };
        }
        if rc.bit(&mut self.choice2) == 0 {
            return match self.mid.get_mut(pos_state) {
                Some(p) => 8usize.wrapping_add(rc.tree(p, 3)),
                None => {
                    rc.bad = true;
                    0
                }
            };
        }
        16usize.wrapping_add(rc.tree(&mut self.high, 8))
    }
}

/// The adaptive probabilities.
struct Model {
    literal: Vec<u16>,
    is_match: [u16; STATES << 4],
    is_rep: [u16; STATES],
    is_rep_g0: [u16; STATES],
    is_rep_g1: [u16; STATES],
    is_rep_g2: [u16; STATES],
    is_rep0_long: [u16; STATES << 4],
    pos_slot: [[u16; 64]; 4],
    pos_special: [u16; 1 + FULL_DISTANCES - END_POS_MODEL],
    align: [u16; 16],
    len: LenModel,
    rep_len: LenModel,
}

impl Model {
    fn new() -> Box<Self> {
        Box::new(Self {
            literal: vec![PROB_INIT; LITERAL_MAX],
            is_match: [PROB_INIT; STATES << 4],
            is_rep: [PROB_INIT; STATES],
            is_rep_g0: [PROB_INIT; STATES],
            is_rep_g1: [PROB_INIT; STATES],
            is_rep_g2: [PROB_INIT; STATES],
            is_rep0_long: [PROB_INIT; STATES << 4],
            pos_slot: [[PROB_INIT; 64]; 4],
            pos_special: [PROB_INIT; 1 + FULL_DISTANCES - END_POS_MODEL],
            align: [PROB_INIT; 16],
            len: LenModel::new(),
            rep_len: LenModel::new(),
        })
    }

    fn reset(&mut self, literal_len: usize) {
        if let Some(literal) = self.literal.get_mut(..literal_len) {
            literal.fill(PROB_INIT);
        }
        self.is_match.fill(PROB_INIT);
        self.is_rep.fill(PROB_INIT);
        self.is_rep_g0.fill(PROB_INIT);
        self.is_rep_g1.fill(PROB_INIT);
        self.is_rep_g2.fill(PROB_INIT);
        self.is_rep0_long.fill(PROB_INIT);
        for slot in &mut self.pos_slot {
            slot.fill(PROB_INIT);
        }
        self.pos_special.fill(PROB_INIT);
        self.align.fill(PROB_INIT);
        self.len = LenModel::new();
        self.rep_len = LenModel::new();
    }

    /// A match distance minus one; `len` is the match length minus
    /// [`MATCH_MIN`].
    fn distance(&mut self, rc: &mut Rc<'_>, len: usize) -> usize {
        let slot = match self.pos_slot.get_mut(len.min(3)) {
            Some(p) => rc.tree(p, 6),
            None => {
                rc.bad = true;
                0
            }
        };
        if slot < 4 {
            return slot;
        }
        // 1 to 30 for slots 4 to 63.
        let direct = u32::try_from(slot >> 1).unwrap_or(0).wrapping_sub(1);
        let base = (2 | (slot & 1)).wrapping_shl(direct);
        if slot < END_POS_MODEL {
            let low = rc.reverse(&mut self.pos_special, base.wrapping_sub(slot), direct);
            base.wrapping_add(low)
        } else {
            let middle = rc.direct(direct.wrapping_sub(ALIGN_BITS));
            let middle = usize::try_from(middle).unwrap_or(0) << ALIGN_BITS;
            let low = rc.reverse(&mut self.align, 0, ALIGN_BITS);
            base.wrapping_add(middle).wrapping_add(low)
        }
    }
}

/// LZMA decoder state carried across the chunks of one block.
struct Lzma {
    m: Box<Model>,
    lc: u32,
    /// `8 - lc`.
    lc_shift: u32,
    lp_mask: usize,
    pb_mask: usize,
    literal_len: usize,
    state: usize,
    /// Distances minus one of the last four matches.
    reps: [usize; 4],
}

impl Lzma {
    fn new() -> Self {
        Self {
            m: Model::new(),
            lc: 0,
            lc_shift: 8,
            lp_mask: 0,
            pb_mask: 0,
            literal_len: LITERAL_CODER,
            state: 0,
            reps: [0; 4],
        }
    }

    /// Sets lc, lp and pb from an LZMA2 properties byte.
    fn set_props(&mut self, props: u8) -> Res<()> {
        if props >= 225 {
            return Err(bad("LZMA properties are out of range"));
        }
        let lc = u32::from(props.wrapping_rem(9));
        let rest = props.wrapping_div(9);
        let lp = u32::from(rest.wrapping_rem(5));
        let pb = u32::from(rest.wrapping_div(5));
        let lc_lp = lc.wrapping_add(lp);
        if lc_lp > 4 {
            return Err(bad("LZMA2 properties have lc + lp over 4"));
        }
        self.lc = lc;
        self.lc_shift = 8u32.wrapping_sub(lc);
        self.lp_mask = 1usize.wrapping_shl(lp).wrapping_sub(1);
        self.pb_mask = 1usize.wrapping_shl(pb).wrapping_sub(1);
        self.literal_len = LITERAL_CODER.wrapping_shl(lc_lp);
        Ok(())
    }

    fn reset(&mut self) {
        self.state = 0;
        self.reps = [0; 4];
        self.m.reset(self.literal_len);
    }

    /// Decodes `out[pos..end]` from `rc`. `dict_start` is the output offset
    /// of the last dictionary reset and `dict_size` the declared dictionary
    /// size; the caller checked `end <= out.len()`.
    fn decode(
        &mut self,
        rc: &mut Rc<'_>,
        out: &mut [u8],
        dict_start: usize,
        mut pos: usize,
        end: usize,
        dict_size: usize,
    ) -> Res<()> {
        let m = &mut *self.m;
        // Bounded: every pass writes at least one byte and `pos` never
        // passes `end`.
        while pos < end {
            let rel = pos.wrapping_sub(dict_start);
            let pos_state = rel & self.pb_mask;
            let state = self.state;
            let ctx = (state << 4) | pos_state;

            if rc.bit_at(&mut m.is_match, ctx) == 0 {
                let prev = if rel == 0 {
                    0
                } else {
                    out.get(pos.wrapping_sub(1)).copied().unwrap_or(0)
                };
                let lit_state = (rel & self.lp_mask).wrapping_shl(self.lc)
                    | usize::from(prev).wrapping_shr(self.lc_shift);
                let base = lit_state.wrapping_mul(LITERAL_CODER);
                let probs = m
                    .literal
                    .get_mut(base..base.wrapping_add(LITERAL_CODER))
                    .ok_or(bad("LZMA literal state is out of range"))?;
                let mut symbol = 1usize;
                if state >= 7 {
                    let at = pos.wrapping_sub(self.reps[0]).wrapping_sub(1);
                    let mut match_byte = usize::from(*out.get(at).ok_or(DIST)?);
                    // Bounded: at most eight bits.
                    while symbol < 0x100 {
                        let match_bit = (match_byte >> 7) & 1;
                        match_byte <<= 1;
                        let bit = rc.bit_at(probs, (0x100usize << match_bit) | symbol);
                        symbol = (symbol << 1) | bit;
                        if match_bit != bit {
                            break;
                        }
                    }
                }
                // Bounded: at most eight bits.
                while symbol < 0x100 {
                    symbol = (symbol << 1) | rc.bit_at(probs, symbol);
                }
                *out.get_mut(pos).ok_or(LONGER)? = u8::try_from(symbol & 0xFF).unwrap_or(0);
                pos = pos.wrapping_add(1);
                self.state = LITERAL_NEXT.get(state).copied().unwrap_or(0);
                continue;
            }

            let len;
            if rc.bit_at(&mut m.is_rep, state) == 1 {
                if rel == 0 {
                    return Err(DIST);
                }
                if rc.bit_at(&mut m.is_rep_g0, state) == 0 {
                    if rc.bit_at(&mut m.is_rep0_long, ctx) == 0 {
                        // Short rep: one byte from rep0.
                        self.state = if state < 7 { 9 } else { 11 };
                        copy_match(out, pos, rel, self.reps[0], 1, end)?;
                        pos = pos.wrapping_add(1);
                        continue;
                    }
                } else {
                    let dist;
                    if rc.bit_at(&mut m.is_rep_g1, state) == 0 {
                        dist = self.reps[1];
                    } else {
                        if rc.bit_at(&mut m.is_rep_g2, state) == 0 {
                            dist = self.reps[2];
                        } else {
                            dist = self.reps[3];
                            self.reps[3] = self.reps[2];
                        }
                        self.reps[2] = self.reps[1];
                    }
                    self.reps[1] = self.reps[0];
                    self.reps[0] = dist;
                }
                len = m.rep_len.decode(rc, pos_state);
                self.state = if state < 7 { 8 } else { 11 };
            } else {
                self.reps[3] = self.reps[2];
                self.reps[2] = self.reps[1];
                self.reps[1] = self.reps[0];
                len = m.len.decode(rc, pos_state);
                self.state = if state < 7 { 7 } else { 10 };
                let dist = m.distance(rc, len);
                // Also rejects the end marker (all ones), which LZMA2 does
                // not allow.
                if dist >= dict_size {
                    return Err(bad("LZMA distance is past the dictionary size"));
                }
                self.reps[0] = dist;
            }
            let len = len.wrapping_add(MATCH_MIN);
            copy_match(out, pos, rel, self.reps[0], len, end)?;
            pos = pos.wrapping_add(len);
        }
        Ok(())
    }
}

/// Copies `len` bytes to `out[pos..]` from `dist + 1` bytes back, where
/// `rel` bytes have been written since the last dictionary reset. The
/// source may overlap the destination.
fn copy_match(
    out: &mut [u8],
    pos: usize,
    rel: usize,
    dist: usize,
    len: usize,
    end: usize,
) -> Res<()> {
    if dist >= rel {
        return Err(DIST);
    }
    if len > end.saturating_sub(pos) {
        return Err(bad("LZMA match runs past its chunk"));
    }
    let back = dist.wrapping_add(1);
    let (head, tail) = out.split_at_mut_checked(pos).ok_or(LONGER)?;
    let src = head.get(pos.wrapping_sub(back)..).ok_or(DIST)?;
    let dst = tail.get_mut(..len).ok_or(LONGER)?;
    if back >= len {
        dst.copy_from_slice(src.get(..len).ok_or(DIST)?);
    } else if let [byte] = *src {
        dst.fill(byte);
    } else {
        // A repeating pattern of `back` bytes: copy it once, then double
        // what is already written.
        dst.get_mut(..back).ok_or(LONGER)?.copy_from_slice(src);
        let mut filled = back;
        // Bounded: `filled` at least doubles until it reaches `len`.
        while filled < len {
            let n = filled.min(len.wrapping_sub(filled));
            let (done, rest) = dst.split_at_mut_checked(filled).ok_or(LONGER)?;
            rest.get_mut(..n)
                .ok_or(LONGER)?
                .copy_from_slice(done.get(..n).ok_or(LONGER)?);
            filled = filled.wrapping_add(n);
        }
    }
    Ok(())
}

// ---- LZMA2 -----------------------------------------------------------------

/// Decodes one block's LZMA2 data at the start of `src` into `out` from
/// `start`. Returns the bytes of `src` used, end marker included, and the
/// output offset where the block's data ends.
fn lzma2(
    lz: &mut Lzma,
    src: &[u8],
    out: &mut [u8],
    start: usize,
    dict_size: usize,
) -> Res<(usize, usize)> {
    let mut ip = 0usize;
    let mut pos = start;
    let mut dict_start = start;
    let mut need_dict_reset = true;
    let mut need_props = true;
    // Bounded: every pass uses at least one byte of `src`.
    loop {
        let control = *src.get(ip).ok_or(TRUNCATED)?;
        ip = ip.saturating_add(1);
        if control == 0x00 {
            return Ok((ip, pos));
        }
        if control >= 0xE0 || control == 0x01 {
            // A dictionary reset; the next LZMA chunk must set properties.
            need_props = true;
            need_dict_reset = false;
            dict_start = pos;
        } else if need_dict_reset {
            return Err(bad("LZMA2 data does not start with a dictionary reset"));
        }

        if control >= 0x80 {
            let unpacked = ((usize::from(control & 0x1F) << 16) | be16(src, ip)?).wrapping_add(1);
            let packed = be16(src, ip.saturating_add(2))?.wrapping_add(1);
            ip = ip.saturating_add(4);
            if control >= 0xC0 {
                let props = *src.get(ip).ok_or(TRUNCATED)?;
                ip = ip.saturating_add(1);
                lz.set_props(props)?;
                lz.reset();
                need_props = false;
            } else if need_props {
                return Err(bad("LZMA2 chunk needs new properties"));
            } else if control >= 0xA0 {
                lz.reset();
            }
            let data = take(src, ip, packed)?;
            ip = ip.saturating_add(packed);
            let end = pos
                .checked_add(unpacked)
                .filter(|&e| e <= out.len())
                .ok_or(LONGER)?;
            let mut rc = Rc::new(data)?;
            lz.decode(&mut rc, out, dict_start, pos, end, dict_size)?;
            if !rc.finished() {
                return Err(bad("LZMA chunk does not end at its packed size"));
            }
            pos = end;
        } else {
            if control > 0x02 {
                return Err(bad("LZMA2 chunk has an invalid control byte"));
            }
            let size = be16(src, ip)?.wrapping_add(1);
            ip = ip.saturating_add(2);
            let data = take(src, ip, size)?;
            ip = ip.saturating_add(size);
            let end = pos.checked_add(size).ok_or(LONGER)?;
            out.get_mut(pos..end).ok_or(LONGER)?.copy_from_slice(data);
            pos = end;
        }
    }
}

// ---- xz container ----------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Check {
    None,
    Crc32,
    Crc64,
    Sha256,
}

impl Check {
    fn of(id: u8) -> Res<Self> {
        match id {
            0x00 => Ok(Self::None),
            0x01 => Ok(Self::Crc32),
            0x04 => Ok(Self::Crc64),
            0x0A => Ok(Self::Sha256),
            _ => Err(bad("xz check type is not supported")),
        }
    }

    fn len(self) -> usize {
        match self {
            Self::None => 0,
            Self::Crc32 => 4,
            Self::Crc64 => 8,
            Self::Sha256 => 32,
        }
    }

    fn verify(self, stored: &[u8], data: &[u8]) -> Res<()> {
        let ok = match self {
            Self::Crc32 => Crc32::of(data).to_le_bytes() == stored,
            Self::Crc64 => crc64(data).to_le_bytes() == stored,
            // Not verified: see the module docs.
            Self::None | Self::Sha256 => true,
        };
        if ok {
            Ok(())
        } else {
            Err(bad("xz block check does not match"))
        }
    }
}

/// A summary of index records, built once from the blocks decoded and once
/// from the index, without storing either list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Records {
    count: u64,
    hash: Crc32,
}

impl Records {
    fn new() -> Self {
        Self {
            count: 0,
            hash: Crc32::new(),
        }
    }

    fn add(&mut self, unpadded: u64, uncompressed: u64) {
        self.count = self.count.saturating_add(1);
        self.hash.update(&unpadded.to_le_bytes());
        self.hash.update(&uncompressed.to_le_bytes());
    }
}

struct BlockHeader {
    len: usize,
    compressed: Option<u64>,
    uncompressed: Option<u64>,
    dict_size: usize,
}

fn block_header(src: &[u8], at: usize) -> Res<BlockHeader> {
    let size_byte = *src.get(at).ok_or(TRUNCATED)?;
    let len = usize::from(size_byte).saturating_add(1).saturating_mul(4);
    let header = take(src, at, len)?;
    let (body, crc) = header
        .split_at_checked(len.saturating_sub(4))
        .ok_or(TRUNCATED)?;
    if Crc32::of(body) != le32(crc, 0)? {
        return Err(bad("xz block header CRC32 does not match"));
    }
    let flags = *body.get(1).ok_or(TRUNCATED)?;
    if flags & 0x3C != 0 {
        return Err(bad("xz block header has reserved flags set"));
    }
    if flags & 0x03 != 0 {
        return Err(bad("xz block has more than one filter"));
    }
    let mut p = 2usize;
    let compressed = if flags & 0x40 != 0 {
        Some(varint(body, &mut p)?)
    } else {
        None
    };
    let uncompressed = if flags & 0x80 != 0 {
        Some(varint(body, &mut p)?)
    } else {
        None
    };
    if varint(body, &mut p)? != 0x21 {
        return Err(bad("xz filter is not LZMA2"));
    }
    if varint(body, &mut p)? != 1 {
        return Err(bad("LZMA2 filter properties are not one byte"));
    }
    let prop = *body.get(p).ok_or(TRUNCATED)?;
    p = p.saturating_add(1);
    if prop > 40 {
        return Err(bad("LZMA2 dictionary size is out of range"));
    }
    let dict_size = if prop == 40 {
        usize::try_from(u32::MAX).unwrap_or(usize::MAX)
    } else {
        (2 | usize::from(prop & 1)).wrapping_shl(u32::from(prop >> 1).wrapping_add(11))
    };
    if body.get(p..).unwrap_or(&[]).iter().any(|&b| b != 0) {
        return Err(bad("xz block header padding is not zero"));
    }
    Ok(BlockHeader {
        len,
        compressed,
        uncompressed,
        dict_size,
    })
}

/// Decodes the xz stream at `start` of `src`, appending to `out` at `*pos`.
/// Returns the offset just past the stream footer.
fn stream(src: &[u8], start: usize, out: &mut [u8], pos: &mut usize, lz: &mut Lzma) -> Res<usize> {
    let header = take(src, start, 12)?;
    let (magic, rest) = header.split_at_checked(6).ok_or(TRUNCATED)?;
    if magic != MAGIC {
        return Err(bad("xz stream has a bad signature"));
    }
    let flags = take(rest, 0, 2)?;
    if Crc32::of(flags) != le32(rest, 2)? {
        return Err(bad("xz stream header CRC32 does not match"));
    }
    let (&[f0, f1], _) = flags.split_first_chunk::<2>().ok_or(TRUNCATED)?;
    if f0 != 0 || f1 & 0xF0 != 0 {
        return Err(bad("xz stream flags are invalid"));
    }
    let check = Check::of(f1)?;
    let mut ip = start.saturating_add(12);

    let mut blocks = Records::new();
    // Bounded: every block uses at least its 8-byte header.
    loop {
        let size_byte = *src.get(ip).ok_or(TRUNCATED)?;
        if size_byte == 0 {
            break;
        }
        let h = block_header(src, ip)?;
        let data_start = ip.saturating_add(h.len);
        let out_start = *pos;
        let (used, end) = lzma2(
            lz,
            src.get(data_start..).ok_or(TRUNCATED)?,
            out,
            out_start,
            h.dict_size,
        )?;
        if h.compressed.is_some_and(|c| c != len64(used)) {
            return Err(bad("xz block compressed size does not match"));
        }
        let produced = end.saturating_sub(out_start);
        if h.uncompressed.is_some_and(|u| u != len64(produced)) {
            return Err(bad("xz block uncompressed size does not match"));
        }
        ip = data_start.saturating_add(used);
        // The header is a multiple of four, so the data alone sets the
        // padding.
        let padding = take(src, ip, pad4(used))?;
        if padding.iter().any(|&b| b != 0) {
            return Err(bad("xz block padding is not zero"));
        }
        ip = ip.saturating_add(padding.len());
        let stored = take(src, ip, check.len())?;
        check.verify(stored, out.get(out_start..end).ok_or(LONGER)?)?;
        ip = ip.saturating_add(check.len());
        let unpadded = h.len.saturating_add(used).saturating_add(check.len());
        blocks.add(len64(unpadded), len64(produced));
        *pos = end;
    }

    // Index.
    let index_start = ip;
    ip = ip.saturating_add(1);
    let count = varint(src, &mut ip)?;
    if count != blocks.count {
        return Err(bad("xz index record count does not match the blocks"));
    }
    let mut index = Records::new();
    // Bounded: `count` equals the number of blocks decoded above.
    for _ in 0..count {
        let unpadded = varint(src, &mut ip)?;
        let uncompressed = varint(src, &mut ip)?;
        index.add(unpadded, uncompressed);
    }
    if index != blocks {
        return Err(bad("xz index does not match the blocks"));
    }
    let padding = take(src, ip, pad4(ip.saturating_sub(index_start)))?;
    if padding.iter().any(|&b| b != 0) {
        return Err(bad("xz index padding is not zero"));
    }
    ip = ip.saturating_add(padding.len());
    let index_crc = le32(src, ip)?;
    if Crc32::of(src.get(index_start..ip).ok_or(TRUNCATED)?) != index_crc {
        return Err(bad("xz index CRC32 does not match"));
    }
    ip = ip.saturating_add(4);
    let index_size = len64(ip.saturating_sub(index_start));

    // Footer.
    let footer = take(src, ip, 12)?;
    if Crc32::of(take(footer, 4, 6)?) != le32(footer, 0)? {
        return Err(bad("xz stream footer CRC32 does not match"));
    }
    let backward = u64::from(le32(footer, 4)?);
    if backward.saturating_add(1).saturating_mul(4) != index_size {
        return Err(bad("xz backward size does not match the index"));
    }
    if take(footer, 8, 2)? != flags {
        return Err(bad("xz stream footer flags do not match the header"));
    }
    if take(footer, 10, 2)? != FOOTER_MAGIC {
        return Err(bad("xz stream footer has a bad signature"));
    }
    Ok(ip.saturating_add(12))
}

/// Decodes the xz data in `src` into exactly `out.len()` bytes.
pub(super) fn xz_decode(src: &[u8], out: &mut [u8]) -> Res<()> {
    let mut lz = Lzma::new();
    let mut pos = 0usize;
    let mut ip = 0usize;
    // Bounded: every pass uses one whole stream (at least 32 bytes) and the
    // padding after it.
    loop {
        ip = stream(src, ip, out, &mut pos, &mut lz)?;
        let padding_start = ip;
        let zeros = src
            .get(ip..)
            .unwrap_or(&[])
            .iter()
            .take_while(|&&b| b == 0)
            .count();
        ip = ip.saturating_add(zeros);
        if pad4(ip.saturating_sub(padding_start)) != 0 {
            return Err(bad("xz stream padding is not a multiple of four"));
        }
        if ip >= src.len() {
            break;
        }
    }
    exact(pos, out.len())
}

#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, unused_imports, missing_docs)]
pub(crate) mod tests {
    use super::*;
    use base64::Engine as _;

    // Fixtures made by liblzma through Python 3.11's `lzma` module from
    // `mixed` below (`repeated` cycles its input up to the length).
    /// xz, CRC64, preset 6, `mixed(4096, 1)`: 1572 bytes.
    const SMALL_XZ: &str = concat!(
        "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",
    );
    /// raw LZMA2, preset 6, `mixed(32768, 2)`: 11165 bytes.
    const MIXED_RAW: &str = concat!(
        "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",
        "GELfrnm3ydDzAAA=",
    );
    /// raw LZMA2, preset 6, `repeated(mixed(8192, 3), 5 MiB)`: 3799 bytes.
    const REPEAT_RAW: &str = concat!(
        "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",
    );
    /// xz, CRC32, lc 0 lp 2 pb 0, `mixed(4096, 4)`: 1624 bytes.
    const PROPS_CRC32_XZ: &str = concat!(
        "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",
    );
    /// xz, SHA-256, lc 4 lp 0 pb 4, `mixed(4096, 5)`: 1648 bytes.
    const PROPS_SHA256_XZ: &str = concat!(
        "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",
    );

    pub(crate) fn fixture(b64: &str) -> Vec<u8> {
        base64::engine::general_purpose::STANDARD
            .decode(b64)
            .unwrap()
    }

    /// The generator the fixtures were made from (mirrors the Python one).
    pub(crate) fn mixed(n: usize, seed: u32) -> Vec<u8> {
        const WORDS: [&[u8]; 12] = [
            b"alpha", b"beta", b"gamma", b"delta", b"disk", b"atlas", b"sector", b"chunk", b"zero",
            b"apple", b"volume", b"index",
        ];
        let mut x = seed;
        let mut out = Vec::with_capacity(n + 512);
        while out.len() < n {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let r = x % 32;
            if r < 12 {
                out.extend_from_slice(WORDS[((x >> 5) % 12) as usize]);
                out.push(b' ');
            } else if r < 24 {
                out.push((x >> 8) as u8);
                out.push((x >> 16) as u8);
            } else if r < 26 && out.len() > 64 {
                let back = 1 + (x >> 5) as usize % out.len().min(40000);
                for _ in 0..3 + (x >> 21) % 40 {
                    out.push(out[out.len() - back]);
                }
            } else if r < 27 {
                out.extend(std::iter::repeat_n(
                    (x >> 24) as u8,
                    1 + ((x >> 8) % 60) as usize,
                ));
            } else {
                out.extend_from_slice(format!("{},", x % 1000).as_bytes());
            }
        }
        out.truncate(n);
        out
    }

    /// An xz stream (CRC64) of liblzma's LZMA2 data and the 32 KiB it
    /// decodes to.
    pub(crate) fn mixed_xz() -> (Vec<u8>, Vec<u8>) {
        let plain = mixed(32768, 2);
        let xz = xz_wrap(&[XzBlock::new(fixture(MIXED_RAW), plain.clone())], 4);
        (xz, plain)
    }

    fn repeat_plain() -> Vec<u8> {
        let unit = mixed(8192, 3);
        unit.iter().copied().cycle().take(5 << 20).collect()
    }

    fn decode(src: &[u8], len: usize) -> Res<Vec<u8>> {
        let mut out = vec![0u8; len];
        xz_decode(src, &mut out).map(|()| out)
    }

    fn varint_bytes(mut v: usize) -> Vec<u8> {
        let mut out = Vec::new();
        loop {
            let b = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    /// A block header body after the size byte: flags, sizes, filters.
    pub(crate) fn lzma2_filter(sizes: Option<(usize, usize)>, dict_prop: u8) -> Vec<u8> {
        let mut h = vec![if sizes.is_some() { 0xC0 } else { 0x00 }];
        if let Some((c, u)) = sizes {
            h.extend(varint_bytes(c));
            h.extend(varint_bytes(u));
        }
        h.extend([0x21, 0x01, dict_prop]);
        h
    }

    pub(crate) struct XzBlock {
        pub lzma2: Vec<u8>,
        pub plain: Vec<u8>,
        /// Header bytes after the size byte, before padding and CRC.
        pub header: Vec<u8>,
    }

    impl XzBlock {
        pub(crate) fn new(lzma2: Vec<u8>, plain: Vec<u8>) -> Self {
            Self {
                lzma2,
                plain,
                header: lzma2_filter(None, 22),
            }
        }
    }

    /// An xz stream around `blocks`, with `check` (0, 1 or 4).
    pub(crate) fn xz_wrap(blocks: &[XzBlock], check: u8) -> Vec<u8> {
        let flags = [0u8, check];
        let mut s = MAGIC.to_vec();
        s.extend(flags);
        s.extend(Crc32::of(&flags).to_le_bytes());
        let mut records = Vec::new();
        for b in blocks {
            let mut h = vec![0u8];
            h.extend(&b.header);
            while (h.len() + 4) % 4 != 0 {
                h.push(0);
            }
            h[0] = ((h.len() + 4) / 4 - 1) as u8;
            let crc = Crc32::of(&h);
            h.extend(crc.to_le_bytes());
            s.extend(&h);
            s.extend(&b.lzma2);
            s.extend(std::iter::repeat_n(0u8, pad4(b.lzma2.len())));
            let chk = match check {
                0 => Vec::new(),
                1 => Crc32::of(&b.plain).to_le_bytes().to_vec(),
                4 => crc64(&b.plain).to_le_bytes().to_vec(),
                _ => panic!("check {check}"),
            };
            s.extend(&chk);
            records.push((h.len() + b.lzma2.len() + chk.len(), b.plain.len()));
        }
        let index_start = s.len();
        s.push(0);
        s.extend(varint_bytes(records.len()));
        for &(u, d) in &records {
            s.extend(varint_bytes(u));
            s.extend(varint_bytes(d));
        }
        s.extend(std::iter::repeat_n(0u8, pad4(s.len() - index_start)));
        let crc = Crc32::of(&s[index_start..]);
        s.extend(crc.to_le_bytes());
        let index_size = s.len() - index_start;
        let mut tail = ((index_size / 4 - 1) as u32).to_le_bytes().to_vec();
        tail.extend(flags);
        s.extend(Crc32::of(&tail).to_le_bytes());
        s.extend(&tail);
        s.extend(FOOTER_MAGIC);
        s
    }

    fn lzma_rs(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        lzma_rs::xz_compress(&mut &data[..], &mut out).unwrap();
        out
    }

    /// The control bytes of the LZMA2 chunks in raw LZMA2 data.
    fn controls(lzma2: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut i = 0;
        loop {
            let c = lzma2[i];
            if c == 0 {
                return out;
            }
            out.push(c);
            i += if c >= 0x80 {
                let packed = u16::from_be_bytes([lzma2[i + 3], lzma2[i + 4]]) as usize + 1;
                5 + usize::from(c >= 0xC0) + packed
            } else {
                3 + u16::from_be_bytes([lzma2[i + 1], lzma2[i + 2]]) as usize + 1
            };
        }
    }

    /// The LZMA2 data inside a one-block xz stream made by lzma-rs or by
    /// [`xz_wrap`] with no sizes.
    fn lzma2_of(xz: &[u8]) -> &[u8] {
        let header = (usize::from(xz[12]) + 1) * 4;
        &xz[12 + header..]
    }

    // ---- range encoder for hand-made streams ----------------------------

    /// A range encoder where every bit uses a fresh probability (1024), as
    /// every context does on its first use.
    struct RcEnc {
        low: u64,
        range: u32,
        cache: u8,
        cache_size: u64,
        out: Vec<u8>,
    }

    impl RcEnc {
        fn new() -> Self {
            Self {
                low: 0,
                range: u32::MAX,
                cache: 0,
                cache_size: 1,
                out: Vec::new(),
            }
        }

        fn shift_low(&mut self) {
            if (self.low as u32) < 0xFF00_0000 || (self.low >> 32) != 0 {
                let carry = (self.low >> 32) as u8;
                let mut temp = self.cache;
                loop {
                    self.out.push(temp.wrapping_add(carry));
                    temp = 0xFF;
                    self.cache_size -= 1;
                    if self.cache_size == 0 {
                        break;
                    }
                }
                self.cache = (self.low >> 24) as u8;
            }
            self.cache_size += 1;
            self.low = (self.low & 0x00FF_FFFF) << 8;
        }

        fn normalize(&mut self) {
            while self.range < TOP {
                self.range <<= 8;
                self.shift_low();
            }
        }

        fn bit(&mut self, bit: u32) {
            let bound = (self.range >> 11) * u32::from(PROB_INIT);
            if bit == 0 {
                self.range = bound;
            } else {
                self.low += u64::from(bound);
                self.range -= bound;
            }
            self.normalize();
        }

        /// A tree symbol, most significant bit first.
        fn bits(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.bit((value >> i) & 1);
            }
        }

        fn reverse(&mut self, value: u32, n: u32) {
            for i in 0..n {
                self.bit((value >> i) & 1);
            }
        }

        fn direct(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                self.range >>= 1;
                if (value >> i) & 1 == 1 {
                    self.low += u64::from(self.range);
                }
                self.normalize();
            }
        }

        fn finish(mut self) -> Vec<u8> {
            for _ in 0..5 {
                self.shift_low();
            }
            self.out
        }
    }

    /// One LZMA chunk (dictionary, state and properties reset; lc 3, lp 0,
    /// pb 2) of `unpacked` bytes around `data`, then the end marker.
    fn lzma_chunk(unpacked: usize, data: &[u8]) -> Vec<u8> {
        let u = unpacked - 1;
        let p = data.len() - 1;
        let mut c = vec![0xE0 | (u >> 16) as u8, (u >> 8) as u8, u as u8];
        c.extend([(p >> 8) as u8, p as u8, 0x5D]);
        c.extend(data);
        c.push(0);
        c
    }

    /// Literal 'A' at position 0 (state 0, prev byte 0, fresh nodes).
    fn literal_a(e: &mut RcEnc) {
        e.bit(0); // is_match[0][0]
        e.bits(u32::from(b'A'), 8);
    }

    /// A simple match at position 1: length 2 (low tree, pos_state 1) and
    /// position slot `slot`.
    fn match_at_1(e: &mut RcEnc, slot: u32) {
        e.bit(1); // is_match[0][1]
        e.bit(0); // is_rep[0]
        e.bit(0); // len choice
        e.bits(0, 3); // low[1]
        e.bits(slot, 6); // pos_slot[0]
    }

    fn crafted(unpacked: usize, enc: RcEnc, dict_prop: u8) -> Vec<u8> {
        let chunk = lzma_chunk(unpacked, &enc.finish());
        let mut block = XzBlock::new(chunk, vec![b'A'; unpacked]);
        block.header = lzma2_filter(None, dict_prop);
        xz_wrap(&[block], 0)
    }

    // ---- tests ------------------------------------------------------------

    #[test]
    fn crc_check_values() {
        assert_eq!(Crc32::of(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc64(b"123456789"), 0x995D_C9BB_DF19_39FA);
        assert_eq!(Crc32::of(b""), 0);
        assert_eq!(crc64(b""), 0);
        // Chunked updates agree with one pass, across the 8-byte path.
        let data = mixed(1000, 9);
        let mut c = Crc32::new();
        for part in data.chunks(13) {
            c.update(part);
        }
        assert_eq!(c.finish(), Crc32::of(&data));
    }

    #[test]
    fn varints() {
        let read = |b: &[u8]| {
            let mut p = 0;
            varint(b, &mut p).map(|v| (v, p))
        };
        assert_eq!(read(&[0x00]), Ok((0, 1)));
        assert_eq!(read(&[0x7F]), Ok((127, 1)));
        assert_eq!(read(&[0x80, 0x01]), Ok((128, 2)));
        assert_eq!(
            read(&[0xFF; 8].iter().copied().chain([0x7F]).collect::<Vec<_>>()),
            Ok((u64::MAX >> 1, 9))
        );
        assert!(read(&[0x80, 0x00]).is_err());
        assert!(read(&[0x80; 9]).is_err());
        assert_eq!(read(&[0x80]), Err(TRUNCATED));
        assert_eq!(read(&[]), Err(TRUNCATED));
    }

    #[test]
    fn generator_matches_the_fixtures() {
        let a = mixed(4096, 1);
        assert!(a.starts_with(b"beta alpha gamma "));
        assert_eq!(a.len(), 4096);
    }

    #[test]
    fn liblzma_xz_streams() {
        for (b64, plain) in [
            (SMALL_XZ, mixed(4096, 1)),
            (PROPS_CRC32_XZ, mixed(4096, 4)),
            (PROPS_SHA256_XZ, mixed(4096, 5)),
        ] {
            let xz = fixture(b64);
            assert_eq!(decode(&xz, plain.len()).unwrap(), plain);
        }
    }

    #[test]
    fn liblzma_raw_lzma2_in_built_streams() {
        let raw = fixture(MIXED_RAW);
        assert_eq!(controls(&raw), [0xE0]);
        let plain = mixed(32768, 2);
        for check in [0, 1, 4] {
            let xz = xz_wrap(&[XzBlock::new(raw.clone(), plain.clone())], check);
            assert_eq!(decode(&xz, plain.len()).unwrap(), plain, "check {check}");
        }
        // Several LZMA chunks that keep the dictionary and state.
        let raw = fixture(REPEAT_RAW);
        let controls = controls(&raw);
        assert!(controls.len() >= 3, "{controls:x?}");
        assert!(controls[0] >= 0xE0);
        assert!(
            controls[1..].iter().all(|&c| (0x80..0xE0).contains(&c)),
            "{controls:x?}"
        );
        let plain = repeat_plain();
        let xz = xz_wrap(&[XzBlock::new(raw, plain.clone())], 4);
        assert_eq!(decode(&xz, plain.len()).unwrap(), plain);
    }

    #[test]
    fn lzma_rs_streams() {
        // lzma-rs writes only uncompressed chunks (0x01, 64 KiB each) with
        // check type none: this exercises the container and the copy path.
        let mut random = vec![0u8; 200_000];
        let mut x = 7u32;
        for b in &mut random {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *b = x as u8;
        }
        for plain in [vec![0x42u8], mixed(1 << 20, 6), vec![7u8; 300_000], random] {
            let xz = lzma_rs(&plain);
            if plain.len() > 0x10000 {
                let c = controls(lzma2_of(&xz));
                assert!(c.len() > 1 && c.iter().all(|&c| c == 0x01), "{c:x?}");
            }
            assert_eq!(decode(&xz, plain.len()).unwrap(), plain);
        }
        assert_eq!(decode(&lzma_rs(&[]), 0).unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn uncompressed_chunks_between_lzma_chunks() {
        let raw = fixture(MIXED_RAW);
        let mut lzma2 = vec![0x01, 0x00, 0x04];
        lzma2.extend(b"hello");
        lzma2.extend([0x02, 0x00, 0x02]);
        lzma2.extend(b"abc");
        lzma2.extend(&raw[..raw.len() - 1]);
        lzma2.extend([0x02, 0x00, 0x00, b'!', 0x00]);
        let mut plain = b"helloabc".to_vec();
        plain.extend(mixed(32768, 2));
        plain.push(b'!');
        let xz = xz_wrap(&[XzBlock::new(lzma2, plain.clone())], 1);
        assert_eq!(decode(&xz, plain.len()).unwrap(), plain);
    }

    #[test]
    fn chunk_sequence_rules() {
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        let run = |lzma2: Vec<u8>, len: usize| {
            decode(&xz_wrap(&[XzBlock::new(lzma2, vec![0; len])], 0), len)
        };
        // First chunk without a dictionary reset.
        assert_eq!(
            run(vec![0x02, 0x00, 0x00, b'x', 0x00], 1),
            Err(bad("LZMA2 data does not start with a dictionary reset"))
        );
        let mut no_reset = raw.clone();
        no_reset[0] = 0xC0;
        assert_eq!(
            run(no_reset, plain.len()),
            Err(bad("LZMA2 data does not start with a dictionary reset"))
        );
        // After a dictionary reset, an LZMA chunk must set properties.
        let mut lzma2 = vec![0x01, 0x00, 0x00, b'x'];
        let mut state_only = raw.clone();
        state_only[0] = 0xA0;
        state_only.remove(5);
        lzma2.extend(&state_only);
        assert_eq!(
            run(lzma2, 1 + plain.len()),
            Err(bad("LZMA2 chunk needs new properties"))
        );
        // Invalid control bytes.
        for c in [0x03u8, 0x10, 0x7F] {
            let lzma2 = vec![0x01, 0x00, 0x00, b'x', c, 0, 0, 0];
            assert_eq!(
                run(lzma2, 2),
                Err(bad("LZMA2 chunk has an invalid control byte"))
            );
        }
        // Missing end marker: the index's zero byte is taken as one, and
        // the stream falls apart after it.
        assert!(run(vec![0x01, 0x00, 0x00, b'x'], 1).is_err());
        let mut lz = Lzma::new();
        let mut out = [0u8; 1];
        assert_eq!(
            super::lzma2(&mut lz, &[0x01, 0x00, 0x00, b'x'], &mut out, 0, 4096),
            Err(TRUNCATED)
        );
    }

    #[test]
    fn bad_properties() {
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        for (props, want) in [
            (225u8, "LZMA properties are out of range"),
            (255, "LZMA properties are out of range"),
            (4 + 9, "LZMA2 properties have lc + lp over 4"),
            (8, "LZMA2 properties have lc + lp over 4"),
            (3 + 2 * 9, "LZMA2 properties have lc + lp over 4"),
        ] {
            let mut r = raw.clone();
            r[5] = props;
            let xz = xz_wrap(&[XzBlock::new(r, plain.clone())], 0);
            assert_eq!(decode(&xz, plain.len()), Err(bad(want)), "{props}");
        }
        // Different but valid properties decode to something else, or fail;
        // never panic.
        for props in [0u8, 4, 9 * 4, 45 * 4 + 4] {
            let mut r = raw.clone();
            r[5] = props;
            let xz = xz_wrap(&[XzBlock::new(r, plain.clone())], 0);
            let _ = decode(&xz, plain.len());
        }
    }

    #[test]
    fn early_ending_chunk_is_rejected() {
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        let packed = u16::from_be_bytes([raw[3], raw[4]]) as usize + 1;
        let data_end = 6 + packed;
        // The attack: declare more packed bytes than the range coder uses,
        // and hide another chunk header in the excess.
        let mut attack = raw[..data_end].to_vec();
        attack.extend([0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x5D, 0x00]);
        attack.extend(&raw[data_end..]);
        let p = packed + 7 - 1;
        attack[3] = (p >> 8) as u8;
        attack[4] = p as u8;
        let xz = xz_wrap(&[XzBlock::new(attack, plain.clone())], 0);
        assert_eq!(
            decode(&xz, plain.len()),
            Err(bad("LZMA chunk does not end at its packed size"))
        );
        // Fewer packed bytes than the coder needs.
        let mut short = raw[..data_end - 1].to_vec();
        short.extend(&raw[data_end..]);
        let p = packed - 2;
        short[3] = (p >> 8) as u8;
        short[4] = p as u8;
        let xz = xz_wrap(&[XzBlock::new(short, plain.clone())], 0);
        assert!(decode(&xz, plain.len()).is_err());
        // Declared unpacked size one larger or smaller than the data.
        for delta in [1i32, -1] {
            let mut r = raw.clone();
            let u = (u16::from_be_bytes([r[1], r[2]]) as i32 + delta) as u16;
            r[1..3].copy_from_slice(&u.to_be_bytes());
            let len = (plain.len() as i32 + delta) as usize;
            let xz = xz_wrap(&[XzBlock::new(r, vec![0; len])], 0);
            assert!(decode(&xz, len).is_err(), "{delta}");
        }
    }

    #[test]
    fn output_size_must_match() {
        let xz = fixture(SMALL_XZ);
        assert_eq!(decode(&xz, 4095), Err(LONGER));
        assert_eq!(decode(&xz, 0), Err(LONGER));
        assert!(matches!(
            decode(&xz, 4097),
            Err(CorruptKind::InvalidValue {
                field: "decoded chunk length",
                value: 4096
            })
        ));
        // A 2 MiB chunk header refused before any decoding.
        let real = lzma_rs(&[0u8; 512]);
        let header_len = 12 + (usize::from(real[12]) + 1) * 4;
        let mut bomb = real[..header_len].to_vec();
        for _ in 0..1000 {
            bomb.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x5D, 0x00]);
        }
        bomb.push(0);
        assert_eq!(decode(&bomb, 512), Err(LONGER));
        // Uncompressed chunks past the end.
        let xz = lzma_rs(&[1u8; 1000]);
        assert_eq!(decode(&xz, 999), Err(LONGER));
    }

    #[test]
    fn hand_made_matches_and_distances() {
        // Positive control: 'A' then a length-2 match at distance 1.
        let mut e = RcEnc::new();
        literal_a(&mut e);
        match_at_1(&mut e, 0);
        assert_eq!(decode(&crafted(3, e, 0), 3).unwrap(), b"AAA");
        // A match at position 0.
        let mut e = RcEnc::new();
        e.bit(1);
        e.bit(0);
        e.bit(0);
        e.bits(0, 3);
        e.bits(0, 6);
        assert_eq!(decode(&crafted(2, e, 0), 2), Err(DIST));
        // A rep match at position 0.
        let mut e = RcEnc::new();
        e.bit(1);
        e.bit(1);
        e.bits(0, 8);
        assert_eq!(decode(&crafted(2, e, 0), 2), Err(DIST));
        // Distance 2 from position 1: before the start.
        let mut e = RcEnc::new();
        literal_a(&mut e);
        match_at_1(&mut e, 1);
        assert_eq!(decode(&crafted(3, e, 0), 3), Err(DIST));
        // A match running past the chunk's unpacked size.
        let mut e = RcEnc::new();
        literal_a(&mut e);
        match_at_1(&mut e, 0);
        assert_eq!(
            decode(&crafted(2, e, 0), 2),
            Err(bad("LZMA match runs past its chunk"))
        );
        // The end marker (distance all ones) is refused even with the
        // largest dictionary.
        let mut e = RcEnc::new();
        literal_a(&mut e);
        match_at_1(&mut e, 63);
        e.direct(u32::MAX, 26);
        e.reverse(0xF, 4);
        assert_eq!(
            decode(&crafted(3, e, 40), 3),
            Err(bad("LZMA distance is past the dictionary size"))
        );
        // A distance past a 4 KiB dictionary in real data.
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        let mut block = XzBlock::new(raw, plain.clone());
        block.header = lzma2_filter(None, 0);
        assert_eq!(
            decode(&xz_wrap(&[block], 0), plain.len()),
            Err(bad("LZMA distance is past the dictionary size"))
        );
    }

    #[test]
    fn truncated_at_every_position() {
        let xz = fixture(SMALL_XZ);
        for n in 0..xz.len() {
            assert!(decode(&xz[..n], 4096).is_err(), "prefix {n}");
        }
        let plain = mixed(4096, 1);
        // Single bit flips: an error, or (where a flip changes nothing that
        // is decoded) the same output.
        for i in 0..xz.len() {
            for bit in [0x01u8, 0x80] {
                let mut x = xz.clone();
                x[i] ^= bit;
                if let Ok(out) = decode(&x, 4096) {
                    assert_eq!(out, plain, "flip at {i}");
                }
            }
        }
    }

    #[test]
    fn container_checks() {
        let xz = fixture(SMALL_XZ);
        let n = xz.len();
        let flip = |at: usize| {
            let mut x = xz.clone();
            x[at] ^= 0x01;
            decode(&x, 4096)
        };
        assert_eq!(flip(0), Err(bad("xz stream has a bad signature")));
        assert_eq!(flip(8), Err(bad("xz stream header CRC32 does not match")));
        assert_eq!(
            flip(12 + 2),
            Err(bad("xz block header CRC32 does not match"))
        );
        assert_eq!(
            flip(n - 12),
            Err(bad("xz stream footer CRC32 does not match"))
        );
        assert_eq!(
            flip(n - 1),
            Err(bad("xz stream footer has a bad signature"))
        );
        // The block check and the index.
        let backward = u32::from_le_bytes(xz[n - 8..n - 4].try_into().unwrap()) as usize;
        let index_start = n - 12 - (backward + 1) * 4;
        assert_eq!(
            flip(index_start - 1),
            Err(bad("xz block check does not match"))
        );
        assert_eq!(
            flip(index_start + 1),
            Err(bad("xz index record count does not match the blocks"))
        );
        assert_eq!(
            flip(index_start + 3),
            Err(bad("xz index does not match the blocks"))
        );
        // Stream flags with a valid CRC but reserved bits, and footer flags
        // that differ from the header.
        let mut x = xz.clone();
        x[6] = 1;
        let crc = Crc32::of(&x[6..8]);
        x[8..12].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(decode(&x, 4096), Err(bad("xz stream flags are invalid")));
        let mut x = xz.clone();
        x[n - 4] = 0x01;
        let crc = Crc32::of(&x[n - 8..n - 2]);
        x[n - 12..n - 8].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            decode(&x, 4096),
            Err(bad("xz stream footer flags do not match the header"))
        );
        // Unsupported check type.
        let mut x = xz.clone();
        x[7] = 0x02;
        let crc = Crc32::of(&x[6..8]);
        x[8..12].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(decode(&x, 4096), Err(bad("xz check type is not supported")));
    }

    #[test]
    fn block_header_rules() {
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        let with_header = |header: Vec<u8>| {
            let mut b = XzBlock::new(raw.clone(), plain.clone());
            b.header = header;
            decode(&xz_wrap(&[b], 1), plain.len())
        };
        // Sizes present and right.
        assert_eq!(
            with_header(lzma2_filter(Some((raw.len(), plain.len())), 22)).unwrap(),
            plain
        );
        assert_eq!(
            with_header(lzma2_filter(Some((raw.len() + 1, plain.len())), 22)),
            Err(bad("xz block compressed size does not match"))
        );
        assert_eq!(
            with_header(lzma2_filter(Some((raw.len(), plain.len() - 1)), 22)),
            Err(bad("xz block uncompressed size does not match"))
        );
        // Wrong filter id (delta, 0x03), two filters, reserved flags.
        assert_eq!(
            with_header(vec![0x00, 0x03, 0x01, 0x00]),
            Err(bad("xz filter is not LZMA2"))
        );
        assert_eq!(
            with_header(vec![0x01, 0x03, 0x01, 0x00, 0x21, 0x01, 22]),
            Err(bad("xz block has more than one filter"))
        );
        assert_eq!(
            with_header(vec![0x04, 0x21, 0x01, 22]),
            Err(bad("xz block header has reserved flags set"))
        );
        // Property size other than one; dictionary property over 40.
        assert_eq!(
            with_header(vec![0x00, 0x21, 0x02, 22, 0]),
            Err(bad("LZMA2 filter properties are not one byte"))
        );
        assert_eq!(
            with_header(vec![0x00, 0x21, 0x01, 41]),
            Err(bad("LZMA2 dictionary size is out of range"))
        );
        // Non-zero header padding.
        assert_eq!(
            with_header(vec![0x00, 0x21, 0x01, 22, 0x00, 0x07]),
            Err(bad("xz block header padding is not zero"))
        );
    }

    #[test]
    fn multiple_blocks_and_streams() {
        let raw = fixture(MIXED_RAW);
        let a = mixed(32768, 2);
        let b = b"second block".to_vec();
        let mut lzma2 = vec![0x01, 0x00, (b.len() - 1) as u8];
        lzma2.extend(&b);
        lzma2.push(0);
        let xz = xz_wrap(
            &[
                XzBlock::new(raw.clone(), a.clone()),
                XzBlock::new(lzma2, b.clone()),
                XzBlock::new(raw.clone(), a.clone()),
            ],
            4,
        );
        let mut plain = a.clone();
        plain.extend(&b);
        plain.extend(&a);
        assert_eq!(decode(&xz, plain.len()).unwrap(), plain);

        // Concatenated streams with padding between and after.
        let s1 = fixture(SMALL_XZ);
        let s2 = fixture(PROPS_CRC32_XZ);
        let mut both = s1.clone();
        both.extend([0u8; 4]);
        both.extend(&s2);
        both.extend([0u8; 8]);
        let mut want = mixed(4096, 1);
        want.extend(mixed(4096, 4));
        assert_eq!(decode(&both, want.len()).unwrap(), want);
        let mut glued = s1.clone();
        glued.extend(&s2);
        assert_eq!(decode(&glued, want.len()).unwrap(), want);
        // Padding not a multiple of four, and trailing garbage.
        for tail in [&[0u8][..], &[0, 0, 0], &[0, 0, 0, 0, 1, 2, 3, 4]] {
            let mut x = s1.clone();
            x.extend(tail);
            assert!(decode(&x, 4096).is_err(), "{tail:?}");
        }
        // An empty stream (no blocks) decodes to nothing.
        assert_eq!(decode(&xz_wrap(&[], 1), 0).unwrap(), Vec::<u8>::new());
    }

    /// Decoded bytes per second through [`xz_decode`]. Run with
    /// `cargo test -p disk-atlas-lib --release xz_decode_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing; run in release with --ignored --nocapture"]
    // Printing the measurement is this test's purpose.
    #[allow(clippy::print_stdout)]
    fn xz_decode_throughput() {
        // 2048 blocks of liblzma output for 32 KiB of mixed data: 64 MiB.
        let raw = fixture(MIXED_RAW);
        let plain = mixed(32768, 2);
        let blocks: Vec<XzBlock> = (0..2048)
            .map(|_| XzBlock::new(raw.clone(), plain.clone()))
            .collect();
        for (check, name) in [(4u8, "CRC64"), (0, "no check")] {
            let xz = xz_wrap(&blocks, check);
            let mut out = vec![0u8; 64 << 20];
            let mut best = std::time::Duration::MAX;
            for _ in 0..4 {
                let start = std::time::Instant::now();
                xz_decode(&xz, &mut out).unwrap();
                best = best.min(start.elapsed());
            }
            assert_eq!(out[..plain.len()], plain[..]);
            let mbps = out.len() as f64 / best.as_secs_f64() / 1e6;
            println!(
                "xz decode: 64 MiB of mixed data ({} MiB compressed, {name}) in {best:?}, {mbps:.0} MB/s",
                xz.len() >> 20
            );
        }
    }
}
