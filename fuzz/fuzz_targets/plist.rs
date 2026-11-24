// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Fuzzes the XML property list parser.
//!
//! Input: the document.

#![no_main]

use atlas_dmg::plist;
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = plist::parse(data);
});
