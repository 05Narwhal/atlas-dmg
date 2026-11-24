// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! Apple disk images (UDIF `.dmg`) and the property list parser they need.
//!
//! [`udif::UdifSource`] presents the disk inside a `.dmg` as a `BlockSource`:
//! chunks stored raw, zero-filled, or compressed with ADC, zlib, bzip2, LZFSE
//! or LZMA (xz) are decoded on demand with bounded memory. [`plist`] is the
//! bounded XML property list parser the image trailer is read with.
//!
//! ```no_run
//! use std::path::Path;
//!
//! use atlas_fs_core::io::block::BlockSource;
//!
//! # fn main() -> Result<(), atlas_fs_core::error::Error> {
//! let disk = atlas_dmg::udif::open(Path::new("installer.dmg"))?;
//! println!("{} bytes of disk", disk.len());
//! # Ok(())
//! # }
//! ```

#![forbid(unsafe_code)]
// On-disk bytes are hostile. These lints make the usual ways of trusting
// them (unchecked indexing, arithmetic and casts, panicking shortcuts) fail
// the clippy gate. Unit tests and the test fixtures are exempt.
#![warn(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unimplemented,
    clippy::unreachable,
    clippy::unwrap_used
)]
#![cfg_attr(
    any(test, feature = "test-util"),
    allow(
        clippy::arithmetic_side_effects,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used
    )
)]

pub mod plist;
pub mod udif;
