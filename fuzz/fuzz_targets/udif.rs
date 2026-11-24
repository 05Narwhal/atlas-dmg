// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes the UDIF reader: the koly trailer, the property list, the mish
//! chunk tables and every chunk decoder behind them.
//!
//! Input: the whole image file.

#![no_main]

use std::sync::Arc;

use atlas_dmg::udif::UdifSource;
use atlas_fs_core::io::block::{BlockSource, MemSource};
use libfuzzer_sys::fuzz_target;

/// Most decoded bytes read per input, so a large claimed disk costs little
/// time.
const READ_LIMIT: u64 = 1 << 20;
/// Bytes per read: one shared cache chunk, plus one byte so reads straddle
/// UDIF chunk boundaries.
const STEP: usize = 64 * 1024 + 1;

fuzz_target!(|data: &[u8]| {
    let file: Arc<dyn BlockSource> = Arc::new(MemSource::new(data));
    let Ok(disk) = UdifSource::new(file) else {
        return;
    };
    let _ = disk.chunk_count();
    let end = disk.len().min(READ_LIMIT);
    let mut buf = vec![0u8; STEP];
    let mut pos = 0;
    while pos < end {
        let n = usize::try_from(end - pos).map_or(STEP, |left| left.min(STEP));
        if disk.read_at(pos, &mut buf[..n]).is_err() {
            break;
        }
        pos += n as u64;
    }
    // Read the start again, after the cache moved on.
    if end > 0 {
        let _ = disk.read_at(0, &mut buf[..1]);
    }
});
