# atlas-dmg

Opens `.dmg` images (UDIF with a `koly` trailer) as a `BlockSource`, decoding zlib, bzip2, LZFSE, LZMA (xz) and ADC chunks on demand with bounded memory. The `plist` module is the bounded XML property list parser the image trailer needs, public because other readers of Apple formats use it too.

- Chunk tables are checked against the file before any chunk is decoded; a damaged table is an error, not a panic.
- LZMA chunks use an in-house decoder with a bounded dictionary.
- No `unsafe`. Every length, offset and count read from the disk is checked; damage is an error, not a panic.

## Licence

Licensed under either of Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT license
([LICENSE-MIT](LICENSE-MIT)) at your option.
