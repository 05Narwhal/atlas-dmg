// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! UDIF `.dmg` images.
//!
//! The last 512 bytes of the file are the `koly` trailer (big-endian). It
//! locates the data fork and an XML property list whose
//! `resource-fork` / `blkx` array holds one `mish` block per partition.
//! Each `mish` block is a table of chunks mapping sector ranges of the disk
//! to byte ranges of the data fork, stored raw, zero-filled or compressed
//! (ADC, zlib, bzip2, LZFSE, or LZMA as an xz stream).
//!
//! [`UdifSource::new`] reads the trailer and the property list once and
//! builds one table of every chunk that covers sectors, sorted by start.
//! Reads binary-search the table. Raw chunks are read straight from the
//! file; compressed chunks are decoded whole on first use and kept in a
//! small LRU cache. Sectors no chunk covers read as zeros.
//!
//! Not supported: segmented images, images without a property list (old
//! resource-fork-only images), version 3 and older trailers. Checksums in
//! the trailer and in `mish` blocks are not verified.
//!
//! Every decoder writes into a buffer of exactly the declared size and
//! fails when the stream would produce more or less; none allocates in
//! proportion to its input or output. xz (LZMA2) chunks use the in-house
//! decoder in [`lzma`], which uses that buffer as its dictionary.

use std::fmt;
use std::io::Read as _;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::plist;
use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};
use atlas_fs_core::io::block::{BlockSource, check_range};
use atlas_fs_core::io::file::FileSource;
use atlas_hfsplus::compression::lzfse_decode;

mod lzma;

/// Bytes per UDIF sector.
const SECTOR: u64 = 512;
/// Size of the `koly` trailer.
const TRAILER_BYTES: usize = 512;
const TRAILER_LEN: u64 = TRAILER_BYTES as u64;
const KOLY: u32 = 0x6B6F_6C79;
const MISH: u32 = 0x6D69_7368;
/// The only trailer version this reader knows.
const VERSION: u32 = 4;
/// Size of a `mish` block before its chunk entries.
const MISH_HEADER: usize = 204;
/// Size of one `mish` chunk entry.
const CHUNK_ENTRY: usize = 40;

/// Largest property list read from an image.
pub const MAX_PLIST_LEN: u64 = 256 << 20;
/// Most chunk entries across all `mish` blocks of one image.
pub const MAX_CHUNKS: u64 = 16 << 20;
/// Largest decoded size of one compressed chunk.
pub const MAX_DECODED_CHUNK: u64 = 64 << 20;
/// Largest stored size of one compressed chunk.
pub const MAX_COMPRESSED_CHUNK: u64 = 64 << 20;
/// Most decoded chunks kept in the cache.
const CACHE_ENTRIES: usize = 8;
/// Most decoded bytes kept in the cache. One chunk larger than this is
/// still kept, alone, so a large chunk is not decoded once per read.
const CACHE_BYTES: usize = 32 << 20;

const TRAILER: &str = "UDIF trailer";
const TABLE: &str = "UDIF block table";
const CHUNK: &str = "UDIF chunk";

/// Chunk types in a `mish` block.
mod kind {
    pub const ZERO: u32 = 0x0000_0000;
    pub const RAW: u32 = 0x0000_0001;
    pub const IGNORE: u32 = 0x0000_0002;
    pub const ADC: u32 = 0x8000_0004;
    pub const ZLIB: u32 = 0x8000_0005;
    pub const BZIP2: u32 = 0x8000_0006;
    pub const LZFSE: u32 = 0x8000_0007;
    pub const LZMA: u32 = 0x8000_0008;
    pub const COMMENT: u32 = 0x7FFF_FFFE;
    pub const TERMINATOR: u32 = 0xFFFF_FFFF;
}

/// Opens the image at `path` read-only.
pub fn open(path: &Path) -> Result<Arc<dyn BlockSource>> {
    let file: Arc<dyn BlockSource> = Arc::new(FileSource::open(path)?);
    Ok(Arc::new(UdifSource::new(file)?))
}

/// How a chunk's sectors are stored.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    /// Zero fill and ignore: reads as zeros, no data.
    Zero,
    Raw,
    Adc,
    Zlib,
    Bzip2,
    Lzfse,
    /// An xz stream.
    Lzma,
}

impl Codec {
    /// The codec for a chunk type; `None` for comment and terminator
    /// entries, which describe no sectors.
    fn of(chunk_type: u32) -> Result<Option<Self>> {
        Ok(Some(match chunk_type {
            kind::ZERO | kind::IGNORE => Self::Zero,
            kind::RAW => Self::Raw,
            kind::ADC => Self::Adc,
            kind::ZLIB => Self::Zlib,
            kind::BZIP2 => Self::Bzip2,
            kind::LZFSE => Self::Lzfse,
            kind::LZMA => Self::Lzma,
            kind::COMMENT | kind::TERMINATOR => return Ok(None),
            other => {
                return Err(Error::unsupported(format!("UDIF chunk type {other:#010x}")));
            }
        }))
    }

    fn is_compressed(self) -> bool {
        !matches!(self, Self::Zero | Self::Raw)
    }
}

/// One entry of the chunk table. Disk positions are bytes, validated at
/// open: `start < end <= disk length`, and for chunks with data
/// `offset + length` lies inside the data fork.
#[derive(Debug, Clone, Copy)]
struct Chunk {
    /// First disk byte.
    start: u64,
    /// One past the last disk byte.
    end: u64,
    /// File offset of the stored data.
    offset: u64,
    /// Stored length in bytes.
    length: u64,
    codec: Codec,
}

/// Decoded chunks, least recently used out.
#[derive(Default)]
struct Cache {
    entries: Vec<CacheEntry>,
    bytes: usize,
    clock: u64,
}

struct CacheEntry {
    index: usize,
    data: Arc<[u8]>,
    used: u64,
}

impl Cache {
    fn get(&mut self, index: usize) -> Option<Arc<[u8]>> {
        self.clock = self.clock.wrapping_add(1);
        let clock = self.clock;
        let entry = self.entries.iter_mut().find(|e| e.index == index)?;
        entry.used = clock;
        Some(Arc::clone(&entry.data))
    }

    fn insert(&mut self, index: usize, data: Arc<[u8]>) {
        if self.entries.iter().any(|e| e.index == index) {
            return;
        }
        // Bounded: every pass removes one entry.
        while !self.entries.is_empty()
            && (self.entries.len() >= CACHE_ENTRIES
                || self.bytes.saturating_add(data.len()) > CACHE_BYTES)
        {
            let oldest = self
                .entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.used)
                .map(|(i, _)| i);
            if let Some(i) = oldest {
                let gone = self.entries.swap_remove(i);
                self.bytes = self.bytes.saturating_sub(gone.data.len());
            }
        }
        self.clock = self.clock.wrapping_add(1);
        self.bytes = self.bytes.saturating_add(data.len());
        self.entries.push(CacheEntry {
            index,
            data,
            used: self.clock,
        });
    }
}

/// The disk inside a UDIF image.
pub struct UdifSource {
    file: Arc<dyn BlockSource>,
    len: u64,
    chunks: Vec<Chunk>,
    cache: Mutex<Cache>,
}

impl fmt::Debug for UdifSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UdifSource")
            .field("len", &self.len)
            .field("chunks", &self.chunks.len())
            .finish_non_exhaustive()
    }
}

fn err(structure: &'static str, at: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(structure, Location::Byte(at), kind))
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn be_u32(b: &[u8], at: usize) -> Option<u32> {
    let bytes = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_be_bytes(bytes.try_into().ok()?))
}

fn be_u64(b: &[u8], at: usize) -> Option<u64> {
    let bytes = b.get(at..at.checked_add(8)?)?;
    Some(u64::from_be_bytes(bytes.try_into().ok()?))
}

/// The trailer fields the reader uses.
struct Trailer {
    /// File offset of the trailer.
    at: u64,
    data_offset: u64,
    data_end: u64,
    xml_offset: u64,
    xml_len: u64,
    sectors: u64,
}

impl Trailer {
    fn read(file: &dyn BlockSource) -> Result<Self> {
        let file_len = file.len();
        let at = file_len.checked_sub(TRAILER_LEN).ok_or_else(|| {
            err(
                TRAILER,
                0,
                CorruptKind::Truncated {
                    needed: TRAILER_LEN,
                    available: file_len,
                },
            )
        })?;
        let mut b = [0u8; TRAILER_BYTES];
        file.read_at(at, &mut b)?;
        let u32_at = |off: usize| be_u32(&b, off).unwrap_or(0);
        let u64_at = |off: usize| be_u64(&b, off).unwrap_or(0);
        let field = |off: u64| at.saturating_add(off);

        let signature = u32_at(0);
        if signature != KOLY {
            return Err(err(
                TRAILER,
                at,
                CorruptKind::BadSignature {
                    expected: u64::from(KOLY),
                    found: u64::from(signature),
                },
            ));
        }
        let version = u32_at(4);
        if version != VERSION {
            return Err(Error::unsupported(format!("UDIF version {version}")));
        }
        let header_size = u32_at(8);
        if u64::from(header_size) != TRAILER_LEN {
            return Err(err(
                TRAILER,
                field(8),
                CorruptKind::InvalidValue {
                    field: "header size",
                    value: u64::from(header_size),
                },
            ));
        }
        if u32_at(60) > 1 {
            return Err(Error::unsupported("segmented disk images"));
        }

        let data_offset = u64_at(24);
        let data_len = u64_at(32);
        let data_end = data_offset.checked_add(data_len).ok_or_else(|| {
            err(
                TRAILER,
                field(24),
                CorruptKind::Overflow {
                    field: "data fork end",
                },
            )
        })?;
        if data_end > at {
            return Err(err(
                TRAILER,
                field(32),
                CorruptKind::OutOfRange {
                    field: "data fork end",
                    value: data_end,
                    limit_name: "trailer offset",
                    limit: at,
                },
            ));
        }

        let xml_offset = u64_at(216);
        let xml_len = u64_at(224);
        if xml_offset == 0 && xml_len == 0 {
            return Err(Error::unsupported("UDIF images without a property list"));
        }
        if xml_len > MAX_PLIST_LEN {
            return Err(err(
                TRAILER,
                field(224),
                CorruptKind::LimitExceeded {
                    what: "property list length",
                    limit: MAX_PLIST_LEN,
                },
            ));
        }
        let xml_end = xml_offset.checked_add(xml_len).ok_or_else(|| {
            err(
                TRAILER,
                field(216),
                CorruptKind::Overflow {
                    field: "property list end",
                },
            )
        })?;
        if xml_end > at {
            return Err(err(
                TRAILER,
                field(224),
                CorruptKind::OutOfRange {
                    field: "property list end",
                    value: xml_end,
                    limit_name: "trailer offset",
                    limit: at,
                },
            ));
        }

        let sectors = u64_at(492);
        if sectors.checked_mul(SECTOR).is_none() {
            return Err(err(
                TRAILER,
                field(492),
                CorruptKind::Overflow {
                    field: "disk length",
                },
            ));
        }

        Ok(Self {
            at,
            data_offset,
            data_end,
            xml_offset,
            xml_len,
            sectors,
        })
    }
}

/// File offset of a chunk's stored data.
///
/// The data fork offset (trailer), plus the `mish` block's data offset,
/// plus the chunk's compressed offset. Images written by hdiutil have the
/// first two at 0, so real images do not yet tell whether the `mish` data
/// offset belongs in this sum; it is kept here, in one place, until
/// conformance testing on real images settles it.
fn chunk_file_offset(data_fork: u64, mish_data: u64, compressed: u64) -> Option<u64> {
    data_fork.checked_add(mish_data)?.checked_add(compressed)
}

/// Builds the chunk table from the `blkx` entries of the property list.
struct TableBuilder<'a> {
    trailer: &'a Trailer,
    chunks: Vec<Chunk>,
    /// Entries seen so far, including those that describe no sectors.
    seen: u64,
}

impl TableBuilder<'_> {
    fn table_err(&self, kind: CorruptKind) -> Error {
        err(TABLE, self.trailer.xml_offset, kind)
    }

    fn mish(&mut self, mish: &[u8]) -> Result<()> {
        let t = self.trailer;
        if mish.len() < MISH_HEADER {
            return Err(self.table_err(CorruptKind::Truncated {
                needed: len64(MISH_HEADER),
                available: len64(mish.len()),
            }));
        }
        let signature = be_u32(mish, 0).unwrap_or(0);
        if signature != MISH {
            return Err(self.table_err(CorruptKind::BadSignature {
                expected: u64::from(MISH),
                found: u64::from(signature),
            }));
        }
        let first_sector = be_u64(mish, 8).unwrap_or(0);
        let block_sectors = be_u64(mish, 16).unwrap_or(0);
        let mish_data = be_u64(mish, 24).unwrap_or(0);
        let count = be_u32(mish, 200).unwrap_or(0);

        let table_end = u64::from(count)
            .checked_mul(len64(CHUNK_ENTRY))
            .and_then(|n| n.checked_add(len64(MISH_HEADER)))
            .filter(|&end| end <= len64(mish.len()))
            .ok_or_else(|| {
                self.table_err(CorruptKind::OutOfRange {
                    field: "chunk table end",
                    value: u64::from(count)
                        .saturating_mul(len64(CHUNK_ENTRY))
                        .saturating_add(len64(MISH_HEADER)),
                    limit_name: "mish block length",
                    limit: len64(mish.len()),
                })
            })?;
        self.seen = self.seen.saturating_add(u64::from(count));
        if self.seen > MAX_CHUNKS {
            return Err(self.table_err(CorruptKind::LimitExceeded {
                what: "UDIF chunk count",
                limit: MAX_CHUNKS,
            }));
        }
        let entries = usize::try_from(table_end)
            .ok()
            .and_then(|end| mish.get(MISH_HEADER..end))
            .unwrap_or(&[]);
        // `count` is bounded by the bytes of the block, checked above.
        self.chunks
            .reserve(usize::try_from(count).unwrap_or(0).min(entries.len()));

        for entry in entries.as_chunks::<CHUNK_ENTRY>().0 {
            let chunk_type = be_u32(entry, 0).unwrap_or(0);
            let Some(codec) = Codec::of(chunk_type)? else {
                continue;
            };
            let sector = be_u64(entry, 8).unwrap_or(0);
            let sectors = be_u64(entry, 16).unwrap_or(0);
            let compressed = be_u64(entry, 24).unwrap_or(0);
            let length = be_u64(entry, 32).unwrap_or(0);
            if sectors == 0 {
                continue;
            }

            let start_sector = first_sector.checked_add(sector).ok_or_else(|| {
                self.table_err(CorruptKind::Overflow {
                    field: "chunk start sector",
                })
            })?;
            let end_sector = start_sector.checked_add(sectors).ok_or_else(|| {
                self.table_err(CorruptKind::Overflow {
                    field: "chunk end sector",
                })
            })?;
            if end_sector > t.sectors {
                return Err(self.table_err(CorruptKind::OutOfRange {
                    field: "chunk end sector",
                    value: end_sector,
                    limit_name: "disk sector count",
                    limit: t.sectors,
                }));
            }
            // And inside its own block. Cannot overflow: the absolute sum
            // above did not.
            let block_end = sector.saturating_add(sectors);
            if block_end > block_sectors {
                return Err(self.table_err(CorruptKind::OutOfRange {
                    field: "chunk end sector in its block",
                    value: block_end,
                    limit_name: "mish sector count",
                    limit: block_sectors,
                }));
            }
            // Cannot overflow: end_sector <= t.sectors, whose byte length
            // was checked with the trailer.
            let overflow = || {
                self.table_err(CorruptKind::Overflow {
                    field: "chunk byte range",
                })
            };
            let start = start_sector.checked_mul(SECTOR).ok_or_else(overflow)?;
            let end = end_sector.checked_mul(SECTOR).ok_or_else(overflow)?;
            let decoded = end.saturating_sub(start);

            let (offset, length) = if codec == Codec::Zero {
                (0, 0)
            } else {
                if codec.is_compressed() {
                    if decoded > MAX_DECODED_CHUNK {
                        return Err(self.table_err(CorruptKind::LimitExceeded {
                            what: "decoded chunk size",
                            limit: MAX_DECODED_CHUNK,
                        }));
                    }
                    if length > MAX_COMPRESSED_CHUNK {
                        return Err(self.table_err(CorruptKind::LimitExceeded {
                            what: "compressed chunk size",
                            limit: MAX_COMPRESSED_CHUNK,
                        }));
                    }
                } else if length < decoded {
                    return Err(self.table_err(CorruptKind::InvalidValue {
                        field: "raw chunk length",
                        value: length,
                    }));
                }
                let offset =
                    chunk_file_offset(t.data_offset, mish_data, compressed).ok_or_else(|| {
                        self.table_err(CorruptKind::Overflow {
                            field: "chunk data offset",
                        })
                    })?;
                // The sum starts at the data fork offset, so the lower bound
                // holds; check the upper one.
                let data_end = offset.checked_add(length).ok_or_else(|| {
                    self.table_err(CorruptKind::Overflow {
                        field: "chunk data end",
                    })
                })?;
                if data_end > t.data_end {
                    return Err(self.table_err(CorruptKind::OutOfRange {
                        field: "chunk data end",
                        value: data_end,
                        limit_name: "data fork end",
                        limit: t.data_end,
                    }));
                }
                (offset, length)
            };

            self.chunks.push(Chunk {
                start,
                end,
                offset,
                length,
                codec,
            });
        }
        Ok(())
    }

    /// Sorts the table and rejects overlaps.
    fn finish(mut self) -> Result<Vec<Chunk>> {
        self.chunks.sort_unstable_by_key(|c| c.start);
        let mut prev_end = 0;
        for c in &self.chunks {
            if c.start < prev_end {
                return Err(self.table_err(CorruptKind::InvalidValue {
                    field: "overlapping chunk start sector",
                    value: c.start / SECTOR,
                }));
            }
            prev_end = c.end;
        }
        Ok(self.chunks)
    }
}

impl UdifSource {
    /// Reads the trailer and the property list of the UDIF image in
    /// `file` and builds the chunk table.
    ///
    /// Fails with [`Error::Unsupported`] for trailer versions other than 4,
    /// segmented images, images without a property list and unknown chunk
    /// types, and with [`Error::Corrupt`] when a field is out of range.
    pub fn new(file: Arc<dyn BlockSource>) -> Result<Self> {
        let trailer = Trailer::read(&*file)?;
        let len = trailer.sectors.saturating_mul(SECTOR);

        let xml_len = usize::try_from(trailer.xml_len).map_err(|_| {
            err(
                TRAILER,
                trailer.at.saturating_add(224),
                CorruptKind::LimitExceeded {
                    what: "property list length",
                    limit: MAX_PLIST_LEN,
                },
            )
        })?;
        let mut xml = vec![0u8; xml_len];
        file.read_at(trailer.xml_offset, &mut xml)?;
        let root = plist::parse(&xml)?;
        drop(xml);

        let blkx = root
            .get("resource-fork")
            .and_then(|fork| fork.get("blkx"))
            .and_then(plist::Value::as_array)
            .ok_or_else(|| {
                err(
                    TABLE,
                    trailer.xml_offset,
                    CorruptKind::Other("property list has no resource-fork blkx array"),
                )
            })?;

        let mut builder = TableBuilder {
            trailer: &trailer,
            chunks: Vec::new(),
            seen: 0,
        };
        for entry in blkx {
            let data = entry
                .get("Data")
                .and_then(plist::Value::as_data)
                .ok_or_else(|| {
                    err(
                        TABLE,
                        trailer.xml_offset,
                        CorruptKind::Other("blkx entry has no Data"),
                    )
                })?;
            builder.mish(data)?;
        }
        let chunks = builder.finish()?;
        drop(root);

        Ok(Self {
            file,
            len,
            chunks,
            cache: Mutex::new(Cache::default()),
        })
    }

    /// Entries in the chunk table: the chunks that cover at least one
    /// sector, comment and terminator entries excluded.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    fn cache(&self) -> MutexGuard<'_, Cache> {
        self.cache.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn read_into(&self, offset: u64, buf: &mut [u8], bulk: bool) -> Result<()> {
        check_range(offset, buf.len(), self.len)?;
        let mut pos = offset;
        let mut rest = buf;
        let mut index = self.chunks.partition_point(|c| c.end <= pos);
        // Bounded: every pass fills at least one byte of `rest`. Chunks are
        // sorted and disjoint, `chunks[index].end > pos` holds on entry and
        // is kept by moving to the next chunk once `pos` reaches the end.
        while !rest.is_empty() {
            let next = self.chunks.get(index);
            let inside = next.filter(|c| c.start <= pos);
            let span = match (inside, next) {
                (Some(c), _) => c.end.saturating_sub(pos),
                (None, Some(c)) => c.start.saturating_sub(pos),
                (None, None) => u64::MAX,
            };
            let n = usize::try_from(span).map_or(rest.len(), |s| s.min(rest.len()));
            let (head, tail) = std::mem::take(&mut rest)
                .split_at_mut_checked(n)
                .filter(|_| n > 0)
                .ok_or_else(|| {
                    err(
                        CHUNK,
                        pos,
                        CorruptKind::Other("chunk table is inconsistent"),
                    )
                })?;
            match inside {
                Some(c) => self.fill(index, c, pos.saturating_sub(c.start), head, bulk)?,
                None => head.fill(0),
            }
            pos = pos.saturating_add(len64(n));
            rest = tail;
            if inside.is_some_and(|c| pos >= c.end) {
                index = index.saturating_add(1);
            }
        }
        Ok(())
    }

    /// Fills `out` from chunk `index`, starting `within` bytes into it.
    fn fill(&self, index: usize, c: &Chunk, within: u64, out: &mut [u8], bulk: bool) -> Result<()> {
        match c.codec {
            Codec::Zero => {
                out.fill(0);
                Ok(())
            }
            Codec::Raw => {
                let at = c.offset.checked_add(within).ok_or_else(|| {
                    err(
                        CHUNK,
                        c.offset,
                        CorruptKind::Overflow {
                            field: "raw chunk read offset",
                        },
                    )
                })?;
                if bulk {
                    self.file.read_bulk(at, out)
                } else {
                    self.file.read_at(at, out)
                }
            }
            _ => {
                let data = self.decoded(index, c)?;
                let src = usize::try_from(within)
                    .ok()
                    .and_then(|start| data.get(start..start.checked_add(out.len())?))
                    .ok_or_else(|| {
                        err(
                            CHUNK,
                            c.offset,
                            CorruptKind::Other("read past the decoded chunk"),
                        )
                    })?;
                out.copy_from_slice(src);
                Ok(())
            }
        }
    }

    /// The decoded bytes of compressed chunk `index`, from the cache or
    /// decoded now. The cache lock is not held while decoding.
    fn decoded(&self, index: usize, c: &Chunk) -> Result<Arc<[u8]>> {
        if let Some(hit) = self.cache().get(index) {
            return Ok(hit);
        }
        let data = self.decode_chunk(c)?;
        self.cache().insert(index, Arc::clone(&data));
        Ok(data)
    }

    fn decode_chunk(&self, c: &Chunk) -> Result<Arc<[u8]>> {
        let too_big = |what| {
            err(
                CHUNK,
                c.offset,
                CorruptKind::LimitExceeded {
                    what,
                    limit: MAX_DECODED_CHUNK,
                },
            )
        };
        // Both were capped at open.
        let decoded = usize::try_from(c.end.saturating_sub(c.start))
            .map_err(|_| too_big("decoded chunk size"))?;
        let stored = usize::try_from(c.length).map_err(|_| too_big("compressed chunk size"))?;
        let mut input = vec![0u8; stored];
        self.file.read_bulk(c.offset, &mut input)?;
        let mut out: Arc<[u8]> = std::iter::repeat_n(0u8, decoded).collect();
        let dst = Arc::get_mut(&mut out).ok_or_else(|| {
            err(
                CHUNK,
                c.offset,
                CorruptKind::Other("decode buffer is shared"),
            )
        })?;
        decode(c.codec, &input, dst).map_err(|kind| err(CHUNK, c.offset, kind))?;
        Ok(out)
    }
}

impl BlockSource for UdifSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn sector_size(&self) -> u32 {
        512
    }

    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_into(offset, buf, false)
    }

    fn read_bulk(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        self.read_into(offset, buf, true)
    }
}

/// Decodes `src` into exactly `out.len()` bytes.
fn decode(codec: Codec, src: &[u8], out: &mut [u8]) -> Result<(), CorruptKind> {
    match codec {
        Codec::Adc => adc(src, out),
        Codec::Zlib => match miniz_oxide::inflate::decompress_slice_iter_to_slice(
            out,
            std::iter::once(src),
            true,
            false,
        ) {
            Ok(n) => exact(n, out.len()),
            // Includes output that would run past `out`: miniz stops at the
            // end of the slice instead of growing anything.
            Err(_) => Err(CorruptKind::Other(
                "zlib data is invalid or longer than its chunk",
            )),
        },
        Codec::Bzip2 => bzip2(src, out),
        Codec::Lzfse => {
            let mut reader = src;
            lzfse_decode(&mut None, &mut reader, out)
        }
        Codec::Lzma => lzma::xz_decode(src, out),
        Codec::Zero | Codec::Raw => Err(CorruptKind::Other("chunk is not compressed")),
    }
}

fn exact(got: usize, want: usize) -> Result<(), CorruptKind> {
    if got == want {
        Ok(())
    } else {
        Err(CorruptKind::InvalidValue {
            field: "decoded chunk length",
            value: len64(got),
        })
    }
}

const LONGER: CorruptKind = CorruptKind::Other("decoded data is longer than its chunk");

/// Apple Data Compression: literal runs and LZ77 matches with byte-by-byte
/// (possibly overlapping) copies.
fn adc(src: &[u8], out: &mut [u8]) -> Result<(), CorruptKind> {
    let truncated = CorruptKind::Other("ADC data ends inside a token");
    let mut ip = 0usize;
    let mut op = 0usize;
    // Bounded: every pass consumes at least one input byte.
    while let Some(&b) = src.get(ip) {
        ip = ip.saturating_add(1);
        if b & 0x80 != 0 {
            let n = usize::from(b & 0x7F).saturating_add(1);
            let lit = src
                .get(ip..ip.saturating_add(n))
                .ok_or_else(|| truncated.clone())?;
            out.get_mut(op..op.saturating_add(n))
                .ok_or(LONGER)?
                .copy_from_slice(lit);
            ip = ip.saturating_add(n);
            op = op.saturating_add(n);
            continue;
        }
        let (len, distance) = if b & 0x40 != 0 {
            let pair = src
                .get(ip..ip.saturating_add(2))
                .ok_or_else(|| truncated.clone())?;
            ip = ip.saturating_add(2);
            let hi = pair.first().copied().unwrap_or(0);
            let lo = pair.get(1).copied().unwrap_or(0);
            (
                usize::from(b & 0x3F).saturating_add(4),
                usize::from(u16::from_be_bytes([hi, lo])),
            )
        } else {
            let lo = *src.get(ip).ok_or_else(|| truncated.clone())?;
            ip = ip.saturating_add(1);
            (
                usize::from((b >> 2) & 0x0F).saturating_add(3),
                (usize::from(b & 0x03) << 8) | usize::from(lo),
            )
        };
        let from = op
            .checked_sub(distance.saturating_add(1))
            .ok_or(CorruptKind::InvalidValue {
                field: "ADC match offset",
                value: len64(distance),
            })?;
        let end = op
            .checked_add(len)
            .filter(|&e| e <= out.len())
            .ok_or(LONGER)?;
        // Byte by byte: the source may overlap the bytes being written.
        for k in 0..len {
            let v = *out.get(from.saturating_add(k)).ok_or(LONGER)?;
            *out.get_mut(op.saturating_add(k)).ok_or(LONGER)? = v;
        }
        op = end;
    }
    exact(op, out.len())
}

fn bzip2(src: &[u8], out: &mut [u8]) -> Result<(), CorruptKind> {
    let invalid = || CorruptKind::Other("bzip2 data is invalid");
    let mut decoder = bzip2::bufread::BzDecoder::new(src);
    let mut filled = 0usize;
    // Bounded: every pass fills at least one byte or stops.
    while let Some(dst) = out.get_mut(filled..).filter(|d| !d.is_empty()) {
        match decoder.read(dst) {
            Ok(0) => break,
            Ok(n) => filled = filled.saturating_add(n),
            Err(_) => return Err(invalid()),
        }
    }
    exact(filled, out.len())?;
    // The stream must end here.
    let mut probe = [0u8; 1];
    match decoder.read(&mut probe) {
        Ok(0) => Ok(()),
        Ok(_) => Err(LONGER),
        Err(_) => Err(invalid()),
    }
}

#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, unused_imports, missing_docs)]
pub mod tests {
    use super::*;
    use atlas_fs_core::io::block::MemSource;
    use base64::Engine as _;
    use std::io::Write as _;

    // ---- encoders -------------------------------------------------------

    /// Bytes that compress, with some variety.
    pub fn pattern(len: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
        (0..len)
            .map(|i| {
                if i % 64 < 40 {
                    b"Disk Atlas reads foreign disks. "[i % 32]
                } else {
                    x ^= x << 13;
                    x ^= x >> 17;
                    x ^= x << 5;
                    (x % 26) as u8 + b'a'
                }
            })
            .collect()
    }

    pub fn zlib(data: &[u8]) -> Vec<u8> {
        miniz_oxide::deflate::compress_to_vec_zlib(data, 6)
    }

    pub fn bzip2(data: &[u8]) -> Vec<u8> {
        let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    pub fn lzfse(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        lzfse_rust::encode_bytes(data, &mut out).unwrap();
        out
    }

    pub fn xz(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        lzma_rs::xz_compress(&mut &data[..], &mut out).unwrap();
        out
    }

    /// ADC of literal runs only.
    pub fn adc_literals(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        for run in data.chunks(128) {
            out.push(0x80 | (run.len() - 1) as u8);
            out.extend_from_slice(run);
        }
        out
    }

    /// Hand-encoded ADC with every token form, including overlapping
    /// copies, and the 512 bytes it decodes to.
    pub fn adc_mixed() -> (Vec<u8>, Vec<u8>) {
        let mut enc = vec![0x83, b'a', b'b', b'c', b'd'];
        let mut want = b"abcd".to_vec();
        // Two-byte match: length 4, offset 3 (back to byte 0).
        enc.extend_from_slice(&[(1 << 2), 3]);
        want.extend_from_slice(b"abcd");
        // Two-byte match, overlapping: length 10, offset 0 (repeat 'd').
        enc.extend_from_slice(&[(7 << 2), 0]);
        want.extend_from_slice(&[b'd'; 10]);
        // Three-byte match, overlapping: length 20, offset 17 (from byte 0).
        enc.extend_from_slice(&[0x40 | (20 - 4), 0, 17]);
        for k in 0..20 {
            let b = want[k];
            want.push(b);
        }
        // Two-byte match with a high offset bit: length 18, offset 256+5.
        let tail = pattern(300, 9);
        enc.extend_from_slice(&adc_literals(&tail));
        want.extend_from_slice(&tail);
        let from = want.len() - 256 - 5 - 1;
        enc.extend_from_slice(&[(15 << 2) | 1, 5]);
        for k in 0..18 {
            let b = want[from + k];
            want.push(b);
        }
        let pad = pattern(512 - want.len(), 10);
        enc.extend_from_slice(&adc_literals(&pad));
        want.extend_from_slice(&pad);
        assert_eq!(want.len(), 512);
        (enc, want)
    }

    // ---- image builder --------------------------------------------------

    /// One `mish` chunk entry and, for chunks with data, what it decodes to.
    #[derive(Clone)]
    pub struct ChunkSpec {
        pub kind: u32,
        /// Relative to the block's first sector.
        pub sector: u64,
        pub sectors: u64,
        pub payload: Vec<u8>,
        pub data: Vec<u8>,
    }

    impl ChunkSpec {
        pub fn coded(kind: u32, sector: u64, data: Vec<u8>, payload: Vec<u8>) -> Self {
            assert_eq!(data.len() % 512, 0);
            Self {
                kind,
                sector,
                sectors: (data.len() / 512) as u64,
                payload,
                data,
            }
        }
        pub fn raw(sector: u64, data: Vec<u8>) -> Self {
            let payload = data.clone();
            Self::coded(kind::RAW, sector, data, payload)
        }
        pub fn zlib(sector: u64, data: Vec<u8>) -> Self {
            let payload = zlib(&data);
            Self::coded(kind::ZLIB, sector, data, payload)
        }
        pub fn bzip2(sector: u64, data: Vec<u8>) -> Self {
            let payload = bzip2(&data);
            Self::coded(kind::BZIP2, sector, data, payload)
        }
        pub fn lzfse(sector: u64, data: Vec<u8>) -> Self {
            let payload = lzfse(&data);
            Self::coded(kind::LZFSE, sector, data, payload)
        }
        pub fn lzma(sector: u64, data: Vec<u8>) -> Self {
            let payload = xz(&data);
            Self::coded(kind::LZMA, sector, data, payload)
        }
        pub fn adc(sector: u64, data: Vec<u8>) -> Self {
            let payload = adc_literals(&data);
            Self::coded(kind::ADC, sector, data, payload)
        }
        pub fn empty(kind: u32, sector: u64, sectors: u64) -> Self {
            Self {
                kind,
                sector,
                sectors,
                payload: Vec::new(),
                data: Vec::new(),
            }
        }
    }

    pub struct Block {
        pub first: u64,
        pub chunks: Vec<ChunkSpec>,
    }

    /// A built image: its bytes, the disk it should present, and the parts
    /// needed to rebuild it with altered `mish` blocks.
    pub struct Built {
        pub bytes: Vec<u8>,
        pub disk: Vec<u8>,
        pub data_fork: Vec<u8>,
        pub mishes: Vec<Vec<u8>>,
        pub sectors: u64,
    }

    impl Built {
        /// The image again with `mishes` in place of the built ones.
        pub fn with_mishes(&self, mishes: &[Vec<u8>]) -> Vec<u8> {
            assemble(&self.data_fork, plist_xml(mishes).as_bytes(), self.sectors)
        }
    }

    /// A `mish` block: entries are (type, sector, sectors, offset, length).
    pub fn mish(first: u64, entries: &[(u32, u64, u64, u64, u64)]) -> Vec<u8> {
        let sectors = entries.iter().map(|e| e.1 + e.2).max().unwrap_or(0);
        let mut m = vec![0u8; MISH_HEADER];
        m[0..4].copy_from_slice(&MISH.to_be_bytes());
        m[4..8].copy_from_slice(&1u32.to_be_bytes());
        m[8..16].copy_from_slice(&first.to_be_bytes());
        m[16..24].copy_from_slice(&sectors.to_be_bytes());
        m[36..40].copy_from_slice(&(entries.len() as u32).to_be_bytes());
        m[200..204].copy_from_slice(&(entries.len() as u32).to_be_bytes());
        for &(t, sector, count, offset, length) in entries {
            m.extend_from_slice(&t.to_be_bytes());
            m.extend_from_slice(&0u32.to_be_bytes());
            m.extend_from_slice(&sector.to_be_bytes());
            m.extend_from_slice(&count.to_be_bytes());
            m.extend_from_slice(&offset.to_be_bytes());
            m.extend_from_slice(&length.to_be_bytes());
        }
        m
    }

    /// The property list hdiutil writes, with base64 broken into
    /// tab-indented lines.
    pub fn plist_xml(mishes: &[Vec<u8>]) -> String {
        let mut x = String::from(super::super::plist::tests::HEADER);
        x.push_str("<plist version=\"1.0\">\n<dict>\n\t<key>resource-fork</key>\n\t<dict>\n");
        x.push_str("\t\t<key>blkx</key>\n\t\t<array>\n");
        for (i, m) in mishes.iter().enumerate() {
            let b64 = base64::engine::general_purpose::STANDARD.encode(m);
            x.push_str(
                "\t\t\t<dict>\n\t\t\t\t<key>Attributes</key>\n\t\t\t\t<string>0x0050</string>\n",
            );
            x.push_str(&format!(
                "\t\t\t\t<key>CFName</key>\n\t\t\t\t<string>Apple_HFS : {i}</string>\n"
            ));
            x.push_str("\t\t\t\t<key>Data</key>\n\t\t\t\t<data>\n");
            for line in b64.as_bytes().chunks(52) {
                x.push_str("\t\t\t\t");
                x.push_str(std::str::from_utf8(line).unwrap());
                x.push('\n');
            }
            x.push_str("\t\t\t\t</data>\n");
            x.push_str(&format!(
                "\t\t\t\t<key>ID</key>\n\t\t\t\t<string>{i}</string>\n\
                 \t\t\t\t<key>Name</key>\n\t\t\t\t<string>disk image (Apple_HFS : {i})</string>\n"
            ));
            x.push_str("\t\t\t</dict>\n");
        }
        x.push_str("\t\t</array>\n\t\t<key>plst</key>\n\t\t<array>\n\t\t\t<dict>\n");
        x.push_str("\t\t\t\t<key>Data</key>\n\t\t\t\t<data>\n\t\t\t\tAAAA\n\t\t\t\t</data>\n");
        x.push_str("\t\t\t</dict>\n\t\t</array>\n\t</dict>\n</dict>\n</plist>\n");
        x
    }

    /// A `koly` trailer.
    pub fn koly(data_len: u64, xml_offset: u64, xml_len: u64, sectors: u64) -> Vec<u8> {
        let mut k = vec![0u8; 512];
        k[0..4].copy_from_slice(&KOLY.to_be_bytes());
        k[4..8].copy_from_slice(&4u32.to_be_bytes());
        k[8..12].copy_from_slice(&512u32.to_be_bytes());
        k[12..16].copy_from_slice(&1u32.to_be_bytes());
        k[32..40].copy_from_slice(&data_len.to_be_bytes());
        k[56..60].copy_from_slice(&1u32.to_be_bytes());
        k[60..64].copy_from_slice(&1u32.to_be_bytes());
        k[216..224].copy_from_slice(&xml_offset.to_be_bytes());
        k[224..232].copy_from_slice(&xml_len.to_be_bytes());
        k[488..492].copy_from_slice(&1u32.to_be_bytes());
        k[492..500].copy_from_slice(&sectors.to_be_bytes());
        k
    }

    /// Data fork, then the property list, then the trailer.
    pub fn assemble(data_fork: &[u8], xml: &[u8], sectors: u64) -> Vec<u8> {
        let mut bytes = data_fork.to_vec();
        bytes.extend_from_slice(xml);
        bytes.extend_from_slice(&koly(
            data_fork.len() as u64,
            data_fork.len() as u64,
            xml.len() as u64,
            sectors,
        ));
        bytes
    }

    pub fn build(blocks: &[Block], sectors: u64) -> Built {
        let mut data_fork = Vec::new();
        let mut disk = vec![0u8; (sectors * 512) as usize];
        let mut mishes = Vec::new();
        for block in blocks {
            let mut entries = Vec::new();
            for c in &block.chunks {
                let offset = data_fork.len() as u64;
                data_fork.extend_from_slice(&c.payload);
                entries.push((c.kind, c.sector, c.sectors, offset, c.payload.len() as u64));
                if !c.data.is_empty() {
                    let at = ((block.first + c.sector) * 512) as usize;
                    disk[at..at + c.data.len()].copy_from_slice(&c.data);
                }
            }
            mishes.push(mish(block.first, &entries));
        }
        let bytes = assemble(&data_fork, plist_xml(&mishes).as_bytes(), sectors);
        Built {
            bytes,
            disk,
            data_fork,
            mishes,
            sectors,
        }
    }

    pub fn open_bytes(bytes: Vec<u8>) -> Result<UdifSource> {
        UdifSource::new(Arc::new(MemSource::new(bytes)))
    }

    fn read_all(src: &UdifSource) -> Vec<u8> {
        let mut out = vec![0u8; src.len() as usize];
        src.read_at(0, &mut out).unwrap();
        out
    }

    fn put_u32(bytes: &mut [u8], at: usize, v: u32) {
        bytes[at..at + 4].copy_from_slice(&v.to_be_bytes());
    }

    fn put_u64(bytes: &mut [u8], at: usize, v: u64) {
        bytes[at..at + 8].copy_from_slice(&v.to_be_bytes());
    }

    fn corruption(r: Result<UdifSource>) -> Corruption {
        match r {
            Err(Error::Corrupt(c)) => c,
            other => panic!("expected corruption, got {other:?}"),
        }
    }

    fn unsupported(r: Result<UdifSource>) -> String {
        match r {
            Err(Error::Unsupported(text)) => text.into_owned(),
            other => panic!("expected unsupported, got {other:?}"),
        }
    }

    /// Every chunk type in one block, with gaps inside and at the end.
    pub fn all_types() -> Built {
        let (adc_enc, adc_want) = adc_mixed();
        let block = Block {
            first: 0,
            chunks: vec![
                ChunkSpec::empty(kind::ZERO, 0, 2),
                ChunkSpec::raw(2, pattern(1024, 1)),
                ChunkSpec::empty(kind::IGNORE, 4, 1),
                ChunkSpec::adc(5, pattern(1024, 2)),
                ChunkSpec::zlib(7, pattern(2048, 3)),
                ChunkSpec::bzip2(11, pattern(1024, 4)),
                ChunkSpec::lzfse(13, pattern(1024, 5)),
                ChunkSpec::lzma(15, pattern(1024, 6)),
                ChunkSpec::empty(kind::COMMENT, 17, 0),
                ChunkSpec::coded(kind::ADC, 20, adc_want, adc_enc),
                ChunkSpec::empty(kind::TERMINATOR, 21, 0),
            ],
        };
        build(&[block], 24)
    }

    /// One compressed chunk at sector 0 of a one-chunk disk.
    fn single(kind: u32, sectors: u64, payload: Vec<u8>) -> Vec<u8> {
        let spec = ChunkSpec {
            kind,
            sector: 0,
            sectors,
            payload,
            data: Vec::new(),
        };
        build(
            &[Block {
                first: 0,
                chunks: vec![spec],
            }],
            sectors.max(1),
        )
        .bytes
    }

    /// The error from reading all of a one-chunk image.
    fn read_error(bytes: Vec<u8>) -> Error {
        let src = open_bytes(bytes).unwrap();
        let mut buf = vec![0u8; src.len() as usize];
        src.read_at(0, &mut buf).unwrap_err()
    }

    fn chunk_corruption(bytes: Vec<u8>) -> Corruption {
        match read_error(bytes) {
            Error::Corrupt(c) => {
                assert_eq!(c.structure, CHUNK);
                c
            }
            other => panic!("expected chunk corruption, got {other:?}"),
        }
    }

    // ---- round trips ----------------------------------------------------

    #[test]
    fn every_chunk_type_round_trips() {
        let built = all_types();
        let src = open_bytes(built.bytes.clone()).unwrap();
        assert_eq!(src.len(), 24 * 512);
        assert_eq!(src.sector_size(), 512);
        assert_eq!(src.chunk_count(), 9);
        assert_eq!(read_all(&src), built.disk);
        // A second pass is served from the cache and still agrees.
        assert_eq!(read_all(&src), built.disk);
        let mut bulk = vec![0u8; built.disk.len()];
        src.read_bulk(0, &mut bulk).unwrap();
        assert_eq!(bulk, built.disk);
    }

    #[test]
    fn reads_straddle_chunks_and_gaps() {
        let built = all_types();
        let src = open_bytes(built.bytes.clone()).unwrap();
        let len = built.disk.len();
        for off in (0..len).step_by(311) {
            for n in [0usize, 1, 2, 511, 512, 513, 1537, 4096] {
                if off + n > len {
                    continue;
                }
                let mut buf = vec![0xEEu8; n];
                src.read_at(off as u64, &mut buf).unwrap();
                assert_eq!(buf, built.disk[off..off + n], "{off}+{n}");
            }
        }
    }

    #[test]
    fn last_byte_and_past_the_end() {
        let built = all_types();
        let src = open_bytes(built.bytes.clone()).unwrap();
        let len = src.len();
        let mut b = [0u8; 1];
        src.read_at(len - 1, &mut b).unwrap();
        assert_eq!(b[0], *built.disk.last().unwrap());
        // The last byte of the last data chunk (the hand-made ADC chunk).
        src.read_at(21 * 512 - 1, &mut b).unwrap();
        assert_eq!(b[0], built.disk[21 * 512 - 1]);
        assert!(src.read_at(len, &mut []).is_ok());
        for (off, n) in [
            (len, 1usize),
            (len - 1, 2),
            (0, len as usize + 1),
            (u64::MAX, 1),
        ] {
            let mut buf = vec![0u8; n];
            assert!(
                matches!(src.read_at(off, &mut buf), Err(Error::OutOfBounds { .. })),
                "{off}+{n}"
            );
            assert!(matches!(
                src.read_bulk(off, &mut buf),
                Err(Error::OutOfBounds { .. })
            ));
        }
    }

    #[test]
    fn multiple_blkx_blocks() {
        let blocks = [
            Block {
                first: 0,
                chunks: vec![
                    ChunkSpec::raw(0, pattern(512, 11)),
                    ChunkSpec::zlib(1, pattern(1536, 12)),
                    ChunkSpec::empty(kind::TERMINATOR, 4, 0),
                ],
            },
            Block {
                first: 10,
                chunks: vec![
                    ChunkSpec::lzfse(0, pattern(1024, 13)),
                    ChunkSpec::empty(kind::ZERO, 2, 3),
                    ChunkSpec::bzip2(5, pattern(512, 14)),
                    ChunkSpec::empty(kind::TERMINATOR, 6, 0),
                ],
            },
            // Listed out of order: the table sorts by start.
            Block {
                first: 6,
                chunks: vec![ChunkSpec::adc(0, pattern(1024, 15))],
            },
        ];
        let built = build(&blocks, 20);
        let src = open_bytes(built.bytes.clone()).unwrap();
        assert_eq!(src.chunk_count(), 6);
        assert_eq!(read_all(&src), built.disk);
        // A read across block boundaries and the gap between them.
        let mut buf = vec![0u8; 9 * 512];
        src.read_at(3 * 512 + 100, &mut buf).unwrap();
        assert_eq!(buf, built.disk[3 * 512 + 100..12 * 512 + 100]);
    }

    #[test]
    fn empty_disk_and_empty_table() {
        let built = build(
            &[Block {
                first: 0,
                chunks: vec![],
            }],
            0,
        );
        let src = open_bytes(built.bytes).unwrap();
        assert_eq!(src.len(), 0);
        assert_eq!(src.chunk_count(), 0);
        assert!(src.read_at(0, &mut []).is_ok());
        // Sectors without chunks read as zeros.
        let built = build(&[], 8);
        let src = open_bytes(built.bytes).unwrap();
        assert_eq!(read_all(&src), vec![0u8; 8 * 512]);
    }

    #[test]
    fn cache_eviction_keeps_reads_correct() {
        // Twelve 1-sector chunks of each codec: more than the cache holds.
        let mut chunks = Vec::new();
        for i in 0..12u64 {
            let data = pattern(512, 100 + i as u32);
            chunks.push(match i % 5 {
                0 => ChunkSpec::zlib(i, data),
                1 => ChunkSpec::bzip2(i, data),
                2 => ChunkSpec::lzfse(i, data),
                3 => ChunkSpec::lzma(i, data),
                _ => ChunkSpec::adc(i, data),
            });
        }
        let built = build(&[Block { first: 0, chunks }], 12);
        let src = open_bytes(built.bytes.clone()).unwrap();
        for round in 0..3 {
            for i in [0usize, 11, 3, 7, 1, 9, 0, 5, 10, 2, 8, 4, 6, 11] {
                let mut buf = vec![0u8; 512];
                src.read_at((i * 512) as u64, &mut buf).unwrap();
                assert_eq!(buf, built.disk[i * 512..(i + 1) * 512], "{round} {i}");
            }
            let cache = src.cache();
            assert!(cache.entries.len() <= CACHE_ENTRIES);
            assert_eq!(
                cache.bytes,
                cache.entries.iter().map(|e| e.data.len()).sum::<usize>()
            );
        }
    }

    #[test]
    fn cache_policy() {
        let mut cache = Cache::default();
        let chunk = |n: usize| -> Arc<[u8]> { vec![0u8; n].into() };
        for i in 0..8 {
            cache.insert(i, chunk(10));
        }
        assert!(cache.get(0).is_some()); // 0 is now the most recent.
        cache.insert(8, chunk(10)); // Evicts 1.
        assert!(cache.get(1).is_none());
        assert!(cache.get(0).is_some());
        assert_eq!(cache.entries.len(), 8);
        // A chunk over the byte cap is kept alone.
        cache.insert(9, chunk(CACHE_BYTES + 1));
        assert_eq!(cache.entries.len(), 1);
        assert!(cache.get(9).is_some());
        cache.insert(10, chunk(16 << 20));
        assert!(cache.get(9).is_none());
        cache.insert(11, chunk(16 << 20));
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.bytes, 32 << 20);
        cache.insert(11, chunk(5));
        assert_eq!(cache.entries.len(), 2);
    }

    #[test]
    fn concurrent_reads_agree() {
        fn assert_bounds<T: Send + Sync>() {}
        assert_bounds::<UdifSource>();
        let built = all_types();
        let src = Arc::new(open_bytes(built.bytes.clone()).unwrap());
        let disk = Arc::new(built.disk);
        std::thread::scope(|s| {
            for t in 0..4usize {
                let src = Arc::clone(&src);
                let disk = Arc::clone(&disk);
                s.spawn(move || {
                    for k in 0..50usize {
                        let off = (k * 977 + t * 131) % (disk.len() - 700);
                        let mut buf = vec![0u8; 700];
                        src.read_at(off as u64, &mut buf).unwrap();
                        assert_eq!(buf, disk[off..off + 700]);
                    }
                });
            }
        });
    }

    #[test]
    fn open_from_a_file() {
        let built = all_types();
        let path =
            std::env::temp_dir().join(format!("disk-atlas-udif-test-{}.dmg", std::process::id()));
        std::fs::write(&path, &built.bytes).unwrap();
        let result = open(&path).map(|src| {
            let mut out = vec![0u8; src.len() as usize];
            src.read_at(0, &mut out).map(|()| out)
        });
        let _ = std::fs::remove_file(&path);
        assert_eq!(result.unwrap().unwrap(), built.disk);
    }

    // ---- trailer --------------------------------------------------------

    fn trailer_at(bytes: &[u8]) -> usize {
        bytes.len() - 512
    }

    #[test]
    fn trailer_signature_and_size() {
        let c = corruption(open_bytes(vec![0u8; 511]));
        assert_eq!(c.structure, TRAILER);
        assert_eq!(
            c.kind,
            CorruptKind::Truncated {
                needed: 512,
                available: 511
            }
        );
        let mut bytes = all_types().bytes;
        let t = trailer_at(&bytes);
        bytes[t] = b'K';
        let c = corruption(open_bytes(bytes));
        assert_eq!(c.location, Location::Byte(t as u64));
        assert!(
            matches!(c.kind, CorruptKind::BadSignature { expected, .. } if expected == u64::from(KOLY))
        );
    }

    #[test]
    fn trailer_version_header_size_and_segments() {
        let base = all_types().bytes;
        let t = trailer_at(&base);
        for version in [0u32, 3, 5] {
            let mut b = base.clone();
            put_u32(&mut b, t + 4, version);
            assert_eq!(
                unsupported(open_bytes(b)),
                format!("UDIF version {version}")
            );
        }
        for size in [0u32, 511, 513, 1024] {
            let mut b = base.clone();
            put_u32(&mut b, t + 8, size);
            let c = corruption(open_bytes(b));
            assert_eq!(c.location, Location::Byte((t + 8) as u64));
            assert!(matches!(
                c.kind,
                CorruptKind::InvalidValue {
                    field: "header size",
                    ..
                }
            ));
        }
        let mut b = base.clone();
        put_u32(&mut b, t + 60, 2);
        assert_eq!(unsupported(open_bytes(b)), "segmented disk images");
        // Segment count 0 (some old images) is accepted.
        let mut b = base.clone();
        put_u32(&mut b, t + 60, 0);
        assert!(open_bytes(b).is_ok());
    }

    #[test]
    fn trailer_data_fork_range() {
        let base = all_types().bytes;
        let t = trailer_at(&base);
        // Data fork running into the trailer.
        let mut b = base.clone();
        put_u64(&mut b, t + 32, t as u64 + 1);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::OutOfRange {
                field: "data fork end",
                ..
            }
        ));
        let mut b = base.clone();
        put_u64(&mut b, t + 24, 1);
        put_u64(&mut b, t + 32, t as u64);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::OutOfRange { .. }
        ));
        // Overflowing end.
        let mut b = base.clone();
        put_u64(&mut b, t + 24, u64::MAX);
        put_u64(&mut b, t + 32, 2);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::Overflow {
                field: "data fork end"
            }
        ));
        // A data fork that ends exactly at the trailer is fine (it covers
        // the property list, which is allowed).
        let mut b = base.clone();
        put_u64(&mut b, t + 32, t as u64);
        assert!(open_bytes(b).is_ok());
    }

    #[test]
    fn trailer_xml_range() {
        let base = all_types().bytes;
        let t = trailer_at(&base);
        let mut b = base.clone();
        put_u64(&mut b, t + 216, 0);
        put_u64(&mut b, t + 224, 0);
        assert_eq!(
            unsupported(open_bytes(b)),
            "UDIF images without a property list"
        );
        let mut b = base.clone();
        put_u64(&mut b, t + 224, MAX_PLIST_LEN + 1);
        assert_eq!(
            corruption(open_bytes(b)).kind,
            CorruptKind::LimitExceeded {
                what: "property list length",
                limit: MAX_PLIST_LEN
            }
        );
        let mut b = base.clone();
        let xml_offset = u64::from_be_bytes(b[t + 216..t + 224].try_into().unwrap());
        put_u64(&mut b, t + 224, t as u64 - xml_offset + 1);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::OutOfRange {
                field: "property list end",
                ..
            }
        ));
        let mut b = base.clone();
        put_u64(&mut b, t + 216, u64::MAX - 3);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::Overflow { .. }
        ));
        // A property list cut short is a property list error.
        let mut b = base.clone();
        let xml_len = u64::from_be_bytes(b[t + 224..t + 232].try_into().unwrap());
        put_u64(&mut b, t + 224, xml_len - 20);
        assert_eq!(corruption(open_bytes(b)).structure, "property list");
    }

    #[test]
    fn trailer_sector_count() {
        let base = all_types().bytes;
        let t = trailer_at(&base);
        let mut b = base.clone();
        put_u64(&mut b, t + 492, u64::MAX / 512 + 1);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::Overflow {
                field: "disk length"
            }
        ));
        // Too few sectors for the chunks.
        let mut b = base.clone();
        put_u64(&mut b, t + 492, 20);
        assert!(matches!(
            corruption(open_bytes(b)).kind,
            CorruptKind::OutOfRange {
                field: "chunk end sector",
                ..
            }
        ));
        // More sectors than the chunks cover: the rest reads as zeros.
        let mut b = base.clone();
        put_u64(&mut b, t + 492, 1 << 40);
        let src = open_bytes(b).unwrap();
        assert_eq!(src.len(), 512 << 40);
        let mut buf = vec![1u8; 4096];
        src.read_at((512 << 40) - 4096, &mut buf).unwrap();
        assert!(buf.iter().all(|&x| x == 0));
    }

    // ---- block table ----------------------------------------------------

    fn table_kind(bytes: Vec<u8>) -> CorruptKind {
        let c = corruption(open_bytes(bytes));
        assert_eq!(c.structure, TABLE, "{c}");
        c.kind
    }

    #[test]
    fn plist_without_blkx() {
        let doc = |body: &str| {
            format!(
                "{}<plist version=\"1.0\">{body}</plist>",
                super::super::plist::tests::HEADER
            )
        };
        for body in [
            "<dict/>",
            "<array/>",
            "<dict><key>resource-fork</key><dict/></dict>",
            "<dict><key>resource-fork</key><dict><key>blkx</key><dict/></dict></dict>",
        ] {
            let bytes = assemble(&[], doc(body).as_bytes(), 0);
            assert!(matches!(table_kind(bytes), CorruptKind::Other(_)), "{body}");
        }
        let body = "<dict><key>resource-fork</key><dict><key>blkx</key><array><dict>\
                    <key>Name</key><string>x</string></dict></array></dict></dict>";
        let bytes = assemble(&[], doc(body).as_bytes(), 0);
        assert_eq!(
            table_kind(bytes),
            CorruptKind::Other("blkx entry has no Data")
        );
        let body = "<dict><key>resource-fork</key><dict><key>blkx</key><array><dict>\
                    <key>Data</key><string>bWlzaA==</string></dict></array></dict></dict>";
        let bytes = assemble(&[], doc(body).as_bytes(), 0);
        assert_eq!(
            table_kind(bytes),
            CorruptKind::Other("blkx entry has no Data")
        );
        // Bad base64 is a property list error.
        let body = "<dict><key>resource-fork</key><dict><key>blkx</key><array><dict>\
                    <key>Data</key><data>@@@@</data></dict></array></dict></dict>";
        let bytes = assemble(&[], doc(body).as_bytes(), 0);
        assert_eq!(corruption(open_bytes(bytes)).structure, "property list");
    }

    #[test]
    fn mish_signature_and_header() {
        let built = all_types();
        let mut m = built.mishes.clone();
        m[0][0] = b'M';
        assert!(matches!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::BadSignature { expected, .. } if expected == u64::from(MISH)
        ));
        let m = vec![built.mishes[0][..203].to_vec()];
        assert_eq!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::Truncated {
                needed: 204,
                available: 203
            }
        );
        let m = vec![Vec::new()];
        assert!(matches!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::Truncated { .. }
        ));
    }

    #[test]
    fn mish_count_overflowing_its_data() {
        let built = all_types();
        let n = (built.mishes[0].len() - 204) / 40;
        for count in [n as u32 + 1, u32::MAX] {
            let mut m = built.mishes.clone();
            put_u32(&mut m[0], 200, count);
            assert!(matches!(
                table_kind(built.with_mishes(&m)),
                CorruptKind::OutOfRange {
                    field: "chunk table end",
                    ..
                }
            ));
        }
        // Fewer than present is fine: trailing bytes are ignored.
        let mut m = built.mishes.clone();
        put_u32(&mut m[0], 200, 2);
        let src = open_bytes(built.with_mishes(&m)).unwrap();
        assert_eq!(src.chunk_count(), 2);
    }

    /// Patches field `at` of chunk entry `entry` in the first `mish`.
    fn patch_entry(built: &Built, entry: usize, at: usize, value: u64) -> Vec<u8> {
        let mut m = built.mishes.clone();
        let base = 204 + entry * 40 + at;
        if at == 0 || at == 4 {
            put_u32(&mut m[0], base, value as u32);
        } else {
            put_u64(&mut m[0], base, value);
        }
        built.with_mishes(&m)
    }

    #[test]
    fn chunk_past_the_disk() {
        let built = all_types();
        // Raw chunk (entry 1) at sector 23 with 2 sectors: ends at 25 > 24.
        let b = patch_entry(&built, 1, 8, 23);
        assert_eq!(
            table_kind(b),
            CorruptKind::OutOfRange {
                field: "chunk end sector",
                value: 25,
                limit_name: "disk sector count",
                limit: 24
            }
        );
        // Sector count overflow.
        let b = patch_entry(&built, 1, 16, u64::MAX);
        assert!(matches!(table_kind(b), CorruptKind::Overflow { .. }));
        // First sector + chunk sector overflow.
        let mut m = built.mishes.clone();
        put_u64(&mut m[0], 8, u64::MAX);
        assert!(matches!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::Overflow { .. }
        ));
        // Zero chunks are bounded too.
        let b = patch_entry(&built, 0, 16, 25);
        assert!(matches!(table_kind(b), CorruptKind::OutOfRange { .. }));
    }

    #[test]
    fn overlapping_chunks() {
        let built = all_types();
        // Raw chunk (sectors 2-3) moved to start at 1, over the zero chunk.
        let b = patch_entry(&built, 1, 8, 1);
        assert_eq!(
            table_kind(b),
            CorruptKind::InvalidValue {
                field: "overlapping chunk start sector",
                value: 1
            }
        );
        // Two blocks covering the same sectors.
        let mut m = built.mishes.clone();
        m.push(built.mishes[0].clone());
        assert!(matches!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::InvalidValue { .. }
        ));
        // Touching is not overlapping: a chunk may start where one ends.
        let m = vec![mish(
            0,
            &[(kind::ZERO, 0, 2, 0, 0), (kind::ZERO, 2, 2, 0, 0)],
        )];
        assert!(open_bytes(built.with_mishes(&m)).is_ok());
    }

    #[test]
    fn chunk_outside_the_data_fork() {
        let built = all_types();
        let fork = built.data_fork.len() as u64;
        // Raw chunk (entry 1) data running one byte past the fork.
        let b = patch_entry(&built, 1, 24, fork - 1023);
        assert!(matches!(
            table_kind(b),
            CorruptKind::OutOfRange {
                field: "chunk data end",
                ..
            }
        ));
        // zlib chunk (entry 4) offset overflowing.
        let b = patch_entry(&built, 4, 24, u64::MAX);
        assert!(matches!(table_kind(b), CorruptKind::Overflow { .. }));
        // mish data offset overflowing the sum.
        let mut m = built.mishes.clone();
        put_u64(&mut m[0], 24, u64::MAX);
        assert!(matches!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::Overflow { .. }
        ));
        // A shrunken data fork in the trailer.
        let mut b = built.bytes.clone();
        let t = trailer_at(&b);
        put_u64(&mut b, t + 32, fork - 1);
        assert!(matches!(
            table_kind(b),
            CorruptKind::OutOfRange {
                field: "chunk data end",
                ..
            }
        ));
        // Zero chunks may point anywhere.
        let b = patch_entry(&built, 0, 24, u64::MAX);
        assert!(open_bytes(b).is_ok());
    }

    #[test]
    fn raw_chunk_shorter_than_its_sectors() {
        let built = all_types();
        let b = patch_entry(&built, 1, 32, 1023);
        assert_eq!(
            table_kind(b),
            CorruptKind::InvalidValue {
                field: "raw chunk length",
                value: 1023
            }
        );
    }

    #[test]
    fn unknown_chunk_types() {
        let built = all_types();
        for t in [3u32, 0x8000_0003, 0x8000_0009, 0x7FFF_FFFD, 0x1234_5678] {
            let b = patch_entry(&built, 1, 0, u64::from(t));
            assert_eq!(
                unsupported(open_bytes(b)),
                format!("UDIF chunk type {t:#010x}")
            );
        }
        // Even when the entry covers no sectors.
        let b = patch_entry(&built, 8, 0, 0x8000_0009);
        assert_eq!(unsupported(open_bytes(b)), "UDIF chunk type 0x80000009");
    }

    #[test]
    fn compressed_size_caps() {
        // Decoded size over the cap.
        let sectors = MAX_DECODED_CHUNK / 512 + 1;
        let m = vec![mish(0, &[(kind::ZLIB, 0, sectors, 0, 10)])];
        let bytes = assemble(&[0u8; 10], plist_xml(&m).as_bytes(), sectors);
        assert_eq!(
            table_kind(bytes),
            CorruptKind::LimitExceeded {
                what: "decoded chunk size",
                limit: MAX_DECODED_CHUNK
            }
        );
        // Exactly at the cap is accepted.
        let sectors = MAX_DECODED_CHUNK / 512;
        let m = vec![mish(0, &[(kind::ZLIB, 0, sectors, 0, 10)])];
        let bytes = assemble(&[0u8; 10], plist_xml(&m).as_bytes(), sectors);
        assert!(open_bytes(bytes).is_ok());
        // Stored size over the cap, checked before the data fork range.
        let m = vec![mish(0, &[(kind::BZIP2, 0, 1, 0, MAX_COMPRESSED_CHUNK + 1)])];
        let bytes = assemble(&[0u8; 10], plist_xml(&m).as_bytes(), 1);
        assert_eq!(
            table_kind(bytes),
            CorruptKind::LimitExceeded {
                what: "compressed chunk size",
                limit: MAX_COMPRESSED_CHUNK
            }
        );
        // Raw chunks are not capped: they are never decoded.
        let sectors = MAX_DECODED_CHUNK / 512 + 1;
        let fork = vec![7u8; (sectors * 512) as usize];
        let m = vec![mish(0, &[(kind::RAW, 0, sectors, 0, sectors * 512)])];
        let bytes = assemble(&fork, plist_xml(&m).as_bytes(), sectors);
        let src = open_bytes(bytes).unwrap();
        let mut buf = [0u8; 3];
        src.read_at(src.len() - 3, &mut buf).unwrap();
        assert_eq!(buf, [7, 7, 7]);
    }

    // ---- codecs ---------------------------------------------------------

    type Encoder = fn(&[u8]) -> Vec<u8>;

    fn codecs() -> [(u32, &'static str, Encoder); 5] {
        [
            (kind::ADC, "adc", adc_literals),
            (kind::ZLIB, "zlib", zlib),
            (kind::BZIP2, "bzip2", bzip2),
            (kind::LZFSE, "lzfse", lzfse),
            (kind::LZMA, "lzma", xz),
        ]
    }

    #[test]
    fn truncated_compressed_data() {
        for (kind, name, encode) in codecs() {
            let payload = encode(&pattern(1024, 21));
            for cut in [1usize, 2, payload.len() / 2] {
                let short = payload[..payload.len() - cut].to_vec();
                let c = chunk_corruption(single(kind, 2, short));
                assert_eq!(c.location, Location::Byte(0), "{name}");
            }
            // Empty payload.
            chunk_corruption(single(kind, 2, Vec::new()));
        }
    }

    #[test]
    fn decoded_size_must_match() {
        for (kind, name, encode) in codecs() {
            // Two sectors of data declared as one: longer.
            let payload = encode(&pattern(1024, 22));
            let c = chunk_corruption(single(kind, 1, payload));
            assert!(
                matches!(
                    c.kind,
                    CorruptKind::Other(_) | CorruptKind::InvalidValue { .. }
                ),
                "{name}: {c}"
            );
            // One sector declared as two: shorter.
            let payload = encode(&pattern(512, 23));
            let c = chunk_corruption(single(kind, 2, payload));
            assert!(
                matches!(
                    c.kind,
                    CorruptKind::Other(_) | CorruptKind::InvalidValue { .. }
                ),
                "{name}: {c}"
            );
            // Not a multiple of the sector size either way.
            let payload = encode(&pattern(700, 24));
            chunk_corruption(single(kind, 2, payload));
        }
    }

    #[test]
    fn garbage_compressed_data() {
        for (kind, name, _) in codecs() {
            for garbage in [vec![0u8; 64], vec![0xFFu8; 64], pattern(300, 25)] {
                let err = read_error(single(kind, 1, garbage));
                assert!(matches!(err, Error::Corrupt(_)), "{name}: {err}");
            }
        }
    }

    #[test]
    fn decompression_bombs_stay_within_the_cap() {
        let zeros = vec![0u8; 16 << 20];
        for (kind, name, encode) in [
            (kind::ZLIB, "zlib", zlib as Encoder),
            (kind::BZIP2, "bzip2", bzip2),
            (kind::LZFSE, "lzfse", lzfse),
        ] {
            let payload = encode(&zeros);
            assert!(payload.len() < 1 << 20, "{name} {}", payload.len());
            let c = chunk_corruption(single(kind, 1, payload));
            assert!(matches!(c.kind, CorruptKind::Other(_)), "{name}: {c}");
        }
        // ADC: one literal then 100k overlapping 18-byte matches (1.8 MB).
        let mut enc = vec![0x80, 0];
        for _ in 0..100_000 {
            enc.extend_from_slice(&[15 << 2, 0]);
        }
        assert_eq!(chunk_corruption(single(kind::ADC, 1, enc)).kind, LONGER);
        // xz: LZMA2 chunk headers declaring 2 MiB each are refused before
        // any decoding.
        let real = xz(&[0u8; 512]);
        let header_len = 12 + (usize::from(real[12]) + 1) * 4;
        let mut bomb = real[..header_len].to_vec();
        for _ in 0..1000 {
            bomb.extend_from_slice(&[0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x5D, 0x00]);
        }
        bomb.push(0);
        assert_eq!(chunk_corruption(single(kind::LZMA, 1, bomb)).kind, LONGER);
    }

    #[test]
    fn adc_tokens() {
        let (enc, want) = adc_mixed();
        let mut out = vec![0u8; 512];
        adc(&enc, &mut out).unwrap();
        assert_eq!(out, want);
        // A match reaching before the start.
        let mut out = vec![0u8; 8];
        assert!(matches!(
            adc(&[0x80, 1, 0x00, 1], &mut out),
            Err(CorruptKind::InvalidValue {
                field: "ADC match offset",
                value: 1
            })
        ));
        assert!(adc(&[0x40, 0, 0], &mut out).is_err());
        // Truncated tokens.
        for enc in [
            &[0x83, 1, 2][..],
            &[0x80, 1, 0x40, 0][..],
            &[0x80, 1, 0x04][..],
        ] {
            assert_eq!(
                adc(enc, &mut out),
                Err(CorruptKind::Other("ADC data ends inside a token"))
            );
        }
        // Literal run past the end.
        let mut out = vec![0u8; 2];
        assert_eq!(adc(&[0x82, 1, 2, 3], &mut out), Err(LONGER));
        // Exact fill.
        assert!(adc(&[0x81, 1, 2], &mut out).is_ok());
        assert_eq!(out, [1, 2]);
        // Empty input, empty output.
        assert!(adc(&[], &mut []).is_ok());
    }

    /// One LZMA chunk of liblzma output: 64 sectors of mixed data.
    fn liblzma_image() -> Built {
        let (payload, data) = lzma::tests::mixed_xz();
        build(
            &[Block {
                first: 2,
                chunks: vec![
                    ChunkSpec::raw(0, pattern(512, 50)),
                    ChunkSpec::coded(kind::LZMA, 1, data, payload),
                    ChunkSpec::empty(kind::TERMINATOR, 65, 0),
                ],
            }],
            70,
        )
    }

    #[test]
    fn liblzma_chunk_round_trips() {
        let built = liblzma_image();
        let src = open_bytes(built.bytes.clone()).unwrap();
        assert_eq!(src.chunk_count(), 2);
        assert_eq!(read_all(&src), built.disk);
        // A read across the raw chunk, the LZMA chunk and the gap after it.
        let mut buf = vec![0u8; 66 * 512];
        src.read_at(2 * 512 + 300, &mut buf).unwrap();
        assert_eq!(buf, built.disk[2 * 512 + 300..68 * 512 + 300]);
    }

    #[test]
    fn chunk_outside_its_block() {
        let built = all_types();
        // The block claims 20 sectors; its last chunk ends at 21.
        let mut m = built.mishes.clone();
        put_u64(&mut m[0], 16, 20);
        assert_eq!(
            table_kind(built.with_mishes(&m)),
            CorruptKind::OutOfRange {
                field: "chunk end sector in its block",
                value: 21,
                limit_name: "mish sector count",
                limit: 20
            }
        );
        // Exactly the block's size is fine.
        put_u64(&mut m[0], 16, 21);
        assert!(open_bytes(built.with_mishes(&m)).is_ok());
        // A second block whose chunk lies inside the disk but past the
        // block's own range.
        let m = vec![
            built.mishes[0].clone(),
            mish(21, &[(kind::ZERO, 0, 2, 0, 0)]),
        ];
        let mut m2 = m.clone();
        put_u64(&mut m2[1], 16, 1);
        assert!(matches!(
            table_kind(built.with_mishes(&m2)),
            CorruptKind::OutOfRange {
                field: "chunk end sector in its block",
                ..
            }
        ));
        assert!(open_bytes(built.with_mishes(&m)).is_ok());
    }

    // ---- fuzz seeds and timing ------------------------------------------

    /// Images for the `udif` fuzz target.
    pub fn seed_images() -> Vec<(&'static str, Vec<u8>)> {
        let multi = build(
            &[
                Block {
                    first: 0,
                    chunks: vec![
                        ChunkSpec::raw(0, pattern(512, 41)),
                        ChunkSpec::zlib(1, pattern(1024, 42)),
                        ChunkSpec::empty(kind::TERMINATOR, 3, 0),
                    ],
                },
                Block {
                    first: 4,
                    chunks: vec![
                        ChunkSpec::empty(kind::ZERO, 0, 2),
                        ChunkSpec::lzma(2, pattern(512, 43)),
                        ChunkSpec::empty(kind::TERMINATOR, 3, 0),
                    ],
                },
            ],
            8,
        );
        let mut seeds = vec![
            ("all_types.dmg", all_types().bytes),
            ("multi_blkx.dmg", multi.bytes),
            ("lzma_liblzma.dmg", liblzma_image().bytes),
        ];
        for (kind, name, encode) in codecs() {
            let one = single(kind, 4, encode(&pattern(2048, 44)));
            seeds.push((name, one));
        }
        seeds
    }

    /// The property lists of the seed images, for the `plist` target.
    pub fn seed_plists() -> Vec<(String, String)> {
        let built = all_types();
        vec![("udif_all_types.plist".into(), plist_xml(&built.mishes))]
    }

    /// Writes the seeds to `$ATLAS_FUZZ_SEEDS/udif`. Does nothing when
    /// the variable is unset. Run it with
    /// `cargo test --lib write_fuzz_seeds -- --ignored`.
    #[test]
    #[ignore = "writes files; run by the fuzz CI job"]
    fn write_fuzz_seeds() {
        let Some(root) = std::env::var_os("ATLAS_FUZZ_SEEDS") else {
            return;
        };
        let dir = Path::new(&root).join("udif");
        std::fs::create_dir_all(&dir).expect("create seed directory");
        for (name, bytes) in seed_images() {
            assert!(bytes.len() <= 1 << 20, "{name} is over the fuzz max_len");
            let src = open_bytes(bytes.clone()).expect("seed opens");
            let mut all = vec![0u8; src.len() as usize];
            src.read_at(0, &mut all).expect("seed reads");
            let file = if name.ends_with(".dmg") {
                name.to_owned()
            } else {
                format!("{name}.dmg")
            };
            std::fs::write(dir.join(file), &bytes).expect("write udif seed");
        }
    }

    /// Decoded bytes per second through a zlib UDIF from memory. Run with
    /// `cargo test -p disk-atlas-lib --release udif_read_throughput -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing; run in release with --ignored --nocapture"]
    // Printing the measurement is this test's purpose.
    #[allow(clippy::print_stdout)]
    fn udif_read_throughput() {
        const LEN: usize = 64 << 20;
        const CHUNK: usize = 1 << 20;
        const READ: usize = 64 << 10;
        let chunks = (0..LEN / CHUNK)
            .map(|i| ChunkSpec::zlib((i * CHUNK / 512) as u64, pattern(CHUNK, i as u32)))
            .collect();
        let built = build(&[Block { first: 0, chunks }], (LEN / 512) as u64);
        let image: Arc<[u8]> = built.bytes.into();
        let mut buf = vec![0u8; READ];
        let mut best = std::time::Duration::MAX;
        // Warm up, then take the best of a few runs, each on a fresh source
        // so every chunk is decoded once per run.
        for _ in 0..4 {
            let src = UdifSource::new(Arc::new(MemSource::new(Arc::clone(&image)))).unwrap();
            let start = std::time::Instant::now();
            let mut pos = 0;
            while pos < LEN {
                src.read_at(pos as u64, &mut buf).unwrap();
                pos += READ;
            }
            best = best.min(start.elapsed());
            assert_eq!(buf[..], built.disk[LEN - READ..]);
        }
        let mbps = LEN as f64 / best.as_secs_f64() / 1e6;
        println!(
            "udif read: 64 MiB zlib image, 1 MiB chunks, 64 KiB reads in {best:?}, {mbps:.0} MB/s (compressed {} MiB)",
            image.len() >> 20
        );
    }
}
