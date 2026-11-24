// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (C) 2026 Julian A. Loy.

//! XML property lists, as found in UDIF images and sparse bundles.
//!
//! [`parse`] reads Apple's XML property list format into a [`Value`] tree.
//! The root element is `<plist>` holding exactly one value. The XML
//! declaration, the DOCTYPE, comments, processing instructions and
//! whitespace between elements are skipped. Entities are never expanded
//! beyond the five predefined ones and character references; any other
//! entity reference is an error, so external entities cannot be reached.
//!
//! Text must be UTF-8. Line ends inside text are normalised to `\n`, as XML
//! requires. CDATA sections are accepted inside text-valued elements.
//!
//! The parser is linear in the input: every event is visited once and text
//! is appended, never re-scanned. Callers cap the input size; nesting is
//! capped at [`MAX_DEPTH`].

use std::borrow::Cow;

use base64::Engine as _;
use base64::alphabet;
use base64::engine::{DecodePaddingMode, GeneralPurpose, GeneralPurposeConfig};
use quick_xml::Reader;
use quick_xml::events::{BytesRef, BytesStart, Event};

use atlas_fs_core::error::{CorruptKind, Corruption, Error, Location, Result};

/// Most nested dicts and arrays, counting the outermost one as 1.
pub const MAX_DEPTH: u64 = 64;

const STRUCTURE: &str = "property list";

/// Standard alphabet; padding may be present or left out, as Apple's own
/// decoder accepts both.
const BASE64: GeneralPurpose = GeneralPurpose::new(
    &alphabet::STANDARD,
    GeneralPurposeConfig::new().with_decode_padding_mode(DecodePaddingMode::Indifferent),
);

/// One property list value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Key and value pairs in document order. Keys may repeat.
    Dict(Vec<(String, Value)>),
    /// Values in document order.
    Array(Vec<Value>),
    /// Unescaped text.
    String(String),
    /// Bytes decoded from base64.
    Data(Vec<u8>),
    /// A signed integer.
    Integer(i128),
    /// A floating point number.
    Real(f64),
    /// `<true/>` or `<false/>`.
    Bool(bool),
    /// An ISO 8601 date, kept as the text found.
    Date(String),
}

impl Value {
    /// The value of the first entry named `key`, when this is a dict.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_dict()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// The entries, when this is a dict.
    pub fn as_dict(&self) -> Option<&[(String, Value)]> {
        match self {
            Self::Dict(entries) => Some(entries),
            _ => None,
        }
    }

    /// The elements, when this is an array.
    pub fn as_array(&self) -> Option<&[Value]> {
        match self {
            Self::Array(items) => Some(items),
            _ => None,
        }
    }

    /// The text, when this is a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(text) => Some(text),
            _ => None,
        }
    }

    /// The bytes, when this is data.
    pub fn as_data(&self) -> Option<&[u8]> {
        match self {
            Self::Data(bytes) => Some(bytes),
            _ => None,
        }
    }

    /// The number, when this is an integer.
    pub fn as_integer(&self) -> Option<i128> {
        match self {
            Self::Integer(n) => Some(*n),
            _ => None,
        }
    }

    /// The number, when this is an integer from 0 to `u64::MAX`.
    pub fn as_u64(&self) -> Option<u64> {
        self.as_integer().and_then(|n| u64::try_from(n).ok())
    }
}

/// Parses an XML property list.
///
/// Fails with [`Error::Corrupt`] (structure "property list", located by
/// byte offset in `xml`) on malformed XML, an unknown element, a dict key
/// without a value, bad base64, an integer that is not a number or does not
/// fit `i128`, text that is not UTF-8, or nesting deeper than
/// [`MAX_DEPTH`].
pub fn parse(xml: &[u8]) -> Result<Value> {
    Parser::new(xml).document()
}

fn corrupt(at: u64, kind: CorruptKind) -> Error {
    Error::Corrupt(Corruption::new(STRUCTURE, Location::Byte(at), kind))
}

fn is_xml_space(b: &u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

/// Element names the format knows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tag {
    Dict,
    Array,
    String,
    Data,
    Integer,
    Real,
    True,
    False,
    Date,
    Key,
}

impl Tag {
    fn of(start: &BytesStart<'_>) -> Option<Self> {
        Some(match start.name().as_ref() {
            b"dict" => Self::Dict,
            b"array" => Self::Array,
            b"string" => Self::String,
            b"data" => Self::Data,
            b"integer" => Self::Integer,
            b"real" => Self::Real,
            b"true" => Self::True,
            b"false" => Self::False,
            b"date" => Self::Date,
            b"key" => Self::Key,
            _ => return None,
        })
    }
}

/// A start tag with its position: `open` is false for `<tag/>`.
struct Open {
    tag: Tag,
    open: bool,
    at: u64,
}

struct Parser<'a> {
    reader: Reader<&'a [u8]>,
}

impl<'a> Parser<'a> {
    fn new(xml: &'a [u8]) -> Self {
        let mut reader = Reader::from_reader(xml);
        let config = reader.config_mut();
        config.check_end_names = true;
        config.expand_empty_elements = false;
        config.trim_text_start = false;
        config.trim_text_end = false;
        config.allow_unmatched_ends = false;
        config.allow_dangling_amp = false;
        Self { reader }
    }

    /// The next event and the byte offset where it starts.
    fn next(&mut self) -> Result<(u64, Event<'a>)> {
        let at = self.reader.buffer_position();
        match self.reader.read_event() {
            Ok(event) => Ok((at, event)),
            Err(_) => Err(corrupt(
                self.reader.error_position(),
                CorruptKind::Other("malformed XML"),
            )),
        }
    }

    /// The next event that is not a comment, a processing instruction or
    /// whitespace-only text. Other text is an error.
    fn next_markup(&mut self) -> Result<(u64, Event<'a>)> {
        // Bounded: every event consumes input, and the reader returns Eof
        // at the end.
        loop {
            let (at, event) = self.next()?;
            match event {
                Event::Comment(_) | Event::PI(_) => {}
                Event::Text(text) if text.iter().all(is_xml_space) => {}
                Event::Text(_) | Event::CData(_) | Event::GeneralRef(_) => {
                    return Err(corrupt(at, CorruptKind::Other("text outside a value")));
                }
                other => return Ok((at, other)),
            }
        }
    }

    fn document(&mut self) -> Result<Value> {
        // Prologue: declaration, DOCTYPE, comments, then <plist>.
        loop {
            let (at, event) = self.next_markup()?;
            match event {
                Event::Decl(_) | Event::DocType(_) => {}
                Event::Start(start) if start.name().as_ref() == b"plist" => break,
                Event::Empty(start) if start.name().as_ref() == b"plist" => {
                    return Err(corrupt(at, CorruptKind::Other("plist holds no value")));
                }
                Event::Eof => {
                    return Err(corrupt(at, CorruptKind::Other("no plist element")));
                }
                _ => return Err(corrupt(at, CorruptKind::Other("root is not plist"))),
            }
        }

        let (at, event) = self.next_markup()?;
        let value = match self.open_of(at, event)? {
            Some(open) => self.value(open, 1)?,
            None => return Err(corrupt(at, CorruptKind::Other("plist holds no value"))),
        };

        let (at, event) = self.next_markup()?;
        match event {
            Event::End(end) if end.name().as_ref() == b"plist" => {}
            _ => {
                return Err(corrupt(
                    at,
                    CorruptKind::Other("plist holds more than one value"),
                ));
            }
        }

        // Epilogue: only comments, processing instructions and whitespace.
        let (at, event) = self.next_markup()?;
        match event {
            Event::Eof => Ok(value),
            _ => Err(corrupt(at, CorruptKind::Other("content after plist"))),
        }
    }

    /// The start tag carried by `event`, `None` for an end tag, an error
    /// for anything else.
    fn open_of(&self, at: u64, event: Event<'a>) -> Result<Option<Open>> {
        let (start, open) = match event {
            Event::Start(start) => (start, true),
            Event::Empty(start) => (start, false),
            Event::End(_) => return Ok(None),
            Event::Eof => {
                return Err(corrupt(at, CorruptKind::Other("unbalanced tags")));
            }
            _ => return Err(corrupt(at, CorruptKind::Other("unexpected markup"))),
        };
        match Tag::of(&start) {
            Some(tag) => Ok(Some(Open { tag, open, at })),
            None => Err(corrupt(at, CorruptKind::Other("unknown element"))),
        }
    }

    /// The value whose start tag was just read, at nesting `depth` (the
    /// root value is 1).
    fn value(&mut self, open: Open, depth: u64) -> Result<Value> {
        let at = open.at;
        match open.tag {
            Tag::Dict | Tag::Array if depth > MAX_DEPTH => Err(corrupt(
                at,
                CorruptKind::LimitExceeded {
                    what: "nesting depth",
                    limit: MAX_DEPTH,
                },
            )),
            Tag::Dict => {
                if open.open {
                    self.dict(depth)
                } else {
                    Ok(Value::Dict(Vec::new()))
                }
            }
            Tag::Array => {
                if open.open {
                    self.array(depth)
                } else {
                    Ok(Value::Array(Vec::new()))
                }
            }
            Tag::Key => Err(corrupt(at, CorruptKind::Other("key outside a dict"))),
            Tag::String => Ok(Value::String(self.text(&open)?)),
            Tag::Date => {
                let text = self.text(&open)?;
                let date = text.trim_matches(|c: char| c.is_ascii_whitespace());
                if date.is_empty() {
                    return Err(corrupt(at, CorruptKind::Other("date is empty")));
                }
                Ok(Value::Date(date.to_owned()))
            }
            Tag::Data => {
                let mut text = self.text(&open)?.into_bytes();
                text.retain(|b| !b.is_ascii_whitespace());
                BASE64
                    .decode(&text)
                    .map(Value::Data)
                    .map_err(|_| corrupt(at, CorruptKind::Other("data is not valid base64")))
            }
            Tag::Integer => parse_integer(&self.text(&open)?)
                .map(Value::Integer)
                .map_err(|kind| corrupt(at, kind)),
            Tag::Real => {
                let text = self.text(&open)?;
                text.trim_matches(|c: char| c.is_ascii_whitespace())
                    .parse::<f64>()
                    .map(Value::Real)
                    .map_err(|_| corrupt(at, CorruptKind::Other("real is not a number")))
            }
            Tag::True | Tag::False => {
                let text = self.text(&open)?;
                if !text.bytes().all(|b| is_xml_space(&b)) {
                    return Err(corrupt(at, CorruptKind::Other("boolean holds text")));
                }
                Ok(Value::Bool(open.tag == Tag::True))
            }
        }
    }

    fn dict(&mut self, depth: u64) -> Result<Value> {
        let mut entries = Vec::new();
        // Bounded: every pass consumes at least one event.
        loop {
            let (at, event) = self.next_markup()?;
            let Some(open) = self.open_of(at, event)? else {
                return Ok(Value::Dict(entries));
            };
            if open.tag != Tag::Key {
                return Err(corrupt(at, CorruptKind::Other("dict entry is not a key")));
            }
            let key = self.text(&open)?;
            let (at, event) = self.next_markup()?;
            let Some(open) = self.open_of(at, event)? else {
                return Err(corrupt(at, CorruptKind::Other("dict key has no value")));
            };
            if open.tag == Tag::Key {
                return Err(corrupt(at, CorruptKind::Other("dict key has no value")));
            }
            let value = self.value(open, depth.saturating_add(1))?;
            entries.push((key, value));
        }
    }

    fn array(&mut self, depth: u64) -> Result<Value> {
        let mut items = Vec::new();
        // Bounded: every pass consumes at least one event.
        loop {
            let (at, event) = self.next_markup()?;
            let Some(open) = self.open_of(at, event)? else {
                return Ok(Value::Array(items));
            };
            items.push(self.value(open, depth.saturating_add(1))?);
        }
    }

    /// The text content of a text-valued element up to its end tag.
    fn text(&mut self, open: &Open) -> Result<String> {
        let mut out = String::new();
        if !open.open {
            return Ok(out);
        }
        // Bounded: every pass consumes at least one event.
        loop {
            let (at, event) = self.next()?;
            match event {
                Event::Text(text) => out.push_str(utf8(at, &text)?),
                Event::CData(data) => out.push_str(utf8(at, &data)?),
                Event::GeneralRef(reference) => out.push(resolve(at, &reference)?),
                Event::Comment(_) | Event::PI(_) => {}
                Event::End(_) => break,
                Event::Eof => {
                    return Err(corrupt(at, CorruptKind::Other("unbalanced tags")));
                }
                Event::Start(_) | Event::Empty(_) | Event::Decl(_) | Event::DocType(_) => {
                    return Err(corrupt(
                        at,
                        CorruptKind::Other("markup inside a text value"),
                    ));
                }
            }
        }
        Ok(match normalise_eols(&out) {
            Cow::Borrowed(_) => out,
            Cow::Owned(normalised) => normalised,
        })
    }
}

fn utf8(at: u64, bytes: &[u8]) -> Result<&str> {
    std::str::from_utf8(bytes).map_err(|_| corrupt(at, CorruptKind::Other("text is not UTF-8")))
}

/// The character a reference such as `&amp;` or `&#x41;` stands for.
fn resolve(at: u64, reference: &BytesRef<'_>) -> Result<char> {
    let unknown = || corrupt(at, CorruptKind::Other("unknown entity reference"));
    match &**reference {
        b"amp" => Ok('&'),
        b"lt" => Ok('<'),
        b"gt" => Ok('>'),
        b"quot" => Ok('"'),
        b"apos" => Ok('\''),
        _ => match reference.resolve_char_ref() {
            Ok(Some(c)) => Ok(c),
            _ => Err(unknown()),
        },
    }
}

/// `\r\n` and lone `\r` become `\n`.
fn normalise_eols(text: &str) -> Cow<'_, str> {
    if !text.contains('\r') {
        return Cow::Borrowed(text);
    }
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() == Some(&'\n') {
                chars.next();
            }
            out.push('\n');
        } else {
            out.push(c);
        }
    }
    Cow::Owned(out)
}

/// Decimal with an optional sign, or hexadecimal after `0x` (also signed),
/// surrounded by optional whitespace.
fn parse_integer(text: &str) -> std::result::Result<i128, CorruptKind> {
    let not_a_number = CorruptKind::Other("integer is not a number");
    let text = text.trim_matches(|c: char| c.is_ascii_whitespace());
    let (negative, rest) = match text.as_bytes().first() {
        Some(b'-') => (true, text.get(1..).unwrap_or("")),
        Some(b'+') => (false, text.get(1..).unwrap_or("")),
        _ => (false, text),
    };
    let (radix, digits) = match rest.get(..2) {
        Some("0x" | "0X") => (16, rest.get(2..).unwrap_or("")),
        _ => (10, rest),
    };
    if digits.is_empty() || !digits.chars().all(|c| c.is_digit(radix)) {
        return Err(not_a_number);
    }
    let overflow = CorruptKind::Overflow { field: "integer" };
    let magnitude = u128::from_str_radix(digits, radix).map_err(|_| overflow.clone())?;
    if negative {
        0i128.checked_sub_unsigned(magnitude).ok_or(overflow)
    } else {
        i128::try_from(magnitude).map_err(|_| overflow)
    }
}

#[cfg(any(test, feature = "test-util"))]
#[doc(hidden)]
#[allow(dead_code, unused_imports, missing_docs)]
pub mod tests {
    use super::*;

    pub const HEADER: &str = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
        <!DOCTYPE plist PUBLIC \"-//Apple//DTD PLIST 1.0//EN\" \
        \"http://www.apple.com/DTDs/PropertyList-1.0.dtd\">\n";

    fn doc(body: &str) -> String {
        format!("{HEADER}<plist version=\"1.0\">\n{body}\n</plist>\n")
    }

    fn ok(body: &str) -> Value {
        parse(doc(body).as_bytes()).unwrap()
    }

    fn err(xml: &[u8]) -> Corruption {
        match parse(xml) {
            Err(Error::Corrupt(c)) => {
                assert_eq!(c.structure, STRUCTURE);
                c
            }
            other => panic!("expected corruption, got {other:?}"),
        }
    }

    fn err_body(body: &str) -> Corruption {
        err(doc(body).as_bytes())
    }

    #[test]
    fn every_value_kind() {
        let v = ok("<dict>
            <key>s</key><string>a &amp; b &lt;c&gt; &#x41;&#66;</string>
            <key>empty</key><string/>
            <key>d</key><data>
                aGVs
                bG8=
            </data>
            <key>i</key><integer>-42</integer>
            <key>h</key><integer>0x1F</integer>
            <key>r</key><real>1.5</real>
            <key>t</key><true/>
            <key>f</key><false></false>
            <key>date</key><date>2026-11-11T00:00:00Z</date>
            <key>a</key><array><integer>1</integer><array/><dict/></array>
        </dict>");
        assert_eq!(v.get("s").unwrap().as_str(), Some("a & b <c> AB"));
        assert_eq!(v.get("empty").unwrap().as_str(), Some(""));
        assert_eq!(v.get("d").unwrap().as_data(), Some(&b"hello"[..]));
        assert_eq!(v.get("i").unwrap().as_integer(), Some(-42));
        assert_eq!(v.get("i").unwrap().as_u64(), None);
        assert_eq!(v.get("h").unwrap().as_u64(), Some(31));
        assert_eq!(v.get("r"), Some(&Value::Real(1.5)));
        assert_eq!(v.get("t"), Some(&Value::Bool(true)));
        assert_eq!(v.get("f"), Some(&Value::Bool(false)));
        assert_eq!(
            v.get("date"),
            Some(&Value::Date("2026-11-11T00:00:00Z".into()))
        );
        let a = v.get("a").unwrap().as_array().unwrap();
        assert_eq!(a.len(), 3);
        assert_eq!(a[1], Value::Array(Vec::new()));
        assert_eq!(a[2], Value::Dict(Vec::new()));
        assert!(v.get("missing").is_none());
        assert!(a[0].get("x").is_none());
        assert!(a[0].as_dict().is_none());
        assert!(v.as_array().is_none());
        assert!(v.as_str().is_none());
        assert!(v.as_data().is_none());
        assert!(v.as_integer().is_none());
    }

    #[test]
    fn dict_keeps_order_and_first_duplicate() {
        let v = ok("<dict><key>b</key><integer>1</integer><key>a</key>\
                    <integer>2</integer><key>b</key><integer>3</integer></dict>");
        let keys: Vec<&str> = v.as_dict().unwrap().iter().map(|(k, _)| &**k).collect();
        assert_eq!(keys, ["b", "a", "b"]);
        assert_eq!(v.get("b").unwrap().as_integer(), Some(1));
    }

    #[test]
    fn prologue_comments_and_whitespace_are_skipped() {
        let xml = "\u{FEFF}<?xml version=\"1.0\"?>\n<!-- c --><?pi x?>\n\
                   <plist><!-- before --><string> keep  spaces </string>\
                   <!-- after --></plist>\n<!-- tail -->\n";
        // A byte-order mark is tolerated by quick-xml or rejected cleanly;
        // without it the document parses.
        let _ = parse(xml.as_bytes());
        let v = parse(xml.trim_start_matches('\u{FEFF}').as_bytes()).unwrap();
        assert_eq!(v.as_str(), Some(" keep  spaces "));
    }

    #[test]
    fn text_details() {
        assert_eq!(ok("<string>a\r\nb\rc</string>").as_str(), Some("a\nb\nc"));
        assert_eq!(
            ok("<string><![CDATA[<x>&]]>y</string>").as_str(),
            Some("<x>&y")
        );
        assert_eq!(ok("<string>a<!-- c -->b</string>").as_str(), Some("ab"));
        assert_eq!(ok("<string>é ü 漢</string>").as_str(), Some("é ü 漢"));
    }

    #[test]
    fn integers() {
        let int = |s: &str| ok(&format!("<integer>{s}</integer>")).as_integer().unwrap();
        assert_eq!(int("0"), 0);
        assert_eq!(int(" +7 "), 7);
        assert_eq!(int("0XfF"), 255);
        assert_eq!(int("-0x10"), -16);
        assert_eq!(int(&u64::MAX.to_string()), i128::from(u64::MAX));
        assert_eq!(int(&i128::MAX.to_string()), i128::MAX);
        assert_eq!(int(&i128::MIN.to_string()), i128::MIN);
        assert_eq!(
            ok(&format!("<integer>{}</integer>", u64::MAX)).as_u64(),
            Some(u64::MAX)
        );
        assert_eq!(
            ok(&format!("<integer>{}</integer>", u128::from(u64::MAX) + 1)).as_u64(),
            None
        );
    }

    #[test]
    fn bad_integers() {
        for s in [
            "", " ", "1.5", "--1", "+-1", "0x", "0xg", "12a", "1 2", "0b1",
        ] {
            let c = err_body(&format!("<integer>{s}</integer>"));
            assert_eq!(
                c.kind,
                CorruptKind::Other("integer is not a number"),
                "{s:?}"
            );
        }
        let huge = "9".repeat(60);
        for s in [huge.as_str(), "170141183460469231731687303715884105728"] {
            let c = err_body(&format!("<integer>{s}</integer>"));
            assert_eq!(c.kind, CorruptKind::Overflow { field: "integer" }, "{s}");
        }
        // i128::MIN - 1.
        let c = err_body("<integer>-170141183460469231731687303715884105729</integer>");
        assert_eq!(c.kind, CorruptKind::Overflow { field: "integer" });
        assert!(matches!(err_body("<integer/>").kind, CorruptKind::Other(_)));
    }

    #[test]
    fn reals_and_dates() {
        assert_eq!(ok("<real> -2.5e3 </real>"), Value::Real(-2500.0));
        assert!(matches!(ok("<real>nan</real>"), Value::Real(r) if r.is_nan()));
        assert_eq!(ok("<real>+infinity</real>"), Value::Real(f64::INFINITY));
        assert!(matches!(
            err_body("<real>abc</real>").kind,
            CorruptKind::Other(_)
        ));
        assert!(matches!(err_body("<real/>").kind, CorruptKind::Other(_)));
        assert!(matches!(err_body("<date/>").kind, CorruptKind::Other(_)));
        assert!(matches!(
            err_body("<true>x</true>").kind,
            CorruptKind::Other(_)
        ));
    }

    #[test]
    fn data_forms() {
        assert_eq!(ok("<data/>").as_data(), Some(&[][..]));
        assert_eq!(ok("<data></data>").as_data(), Some(&[][..]));
        assert_eq!(ok("<data>aGVsbG8</data>").as_data(), Some(&b"hello"[..]));
        assert_eq!(
            ok("<data>\n\taGVs\n\tbG8=\n</data>").as_data(),
            Some(&b"hello"[..])
        );
        for bad in ["a", "aGVsbG8=!", "aGVs-G8=", "aGVsbG8==", "a===", "aGV_"] {
            let c = err_body(&format!("<data>{bad}</data>"));
            assert_eq!(
                c.kind,
                CorruptKind::Other("data is not valid base64"),
                "{bad}"
            );
        }
    }

    #[test]
    fn nesting_depth() {
        let nest = |n: usize| {
            format!(
                "{}<string>x</string>{}",
                "<array>".repeat(n),
                "</array>".repeat(n)
            )
        };
        let v = ok(&nest(64));
        let mut cur = &v;
        for _ in 0..63 {
            cur = &cur.as_array().unwrap()[0];
        }
        assert_eq!(cur.as_array().unwrap()[0].as_str(), Some("x"));
        let c = err_body(&nest(65));
        assert_eq!(
            c.kind,
            CorruptKind::LimitExceeded {
                what: "nesting depth",
                limit: MAX_DEPTH
            }
        );
        // Dicts count too, and empty forms count as containers.
        let dicts = format!(
            "{}<dict/>{}",
            "<dict><key>k</key>".repeat(64),
            "</dict>".repeat(64)
        );
        assert!(matches!(
            err_body(&dicts).kind,
            CorruptKind::LimitExceeded { .. }
        ));
        // Very deep input fails fast without exhausting the stack.
        assert!(matches!(
            err_body(&nest(100_000)).kind,
            CorruptKind::LimitExceeded { .. }
        ));
    }

    #[test]
    fn structural_errors() {
        // Key without a value, at the end and followed by another key.
        for body in [
            "<dict><key>a</key></dict>",
            "<dict><key>a</key><key>b</key><string/></dict>",
            "<dict><string>a</string></dict>",
            "<dict><key>a</key><key/></dict>",
            "<array><key>a</key></array>",
            "<dict>text<key>a</key><true/></dict>",
            "<array><string>a</string>stray</array>",
        ] {
            assert!(
                matches!(err_body(body).kind, CorruptKind::Other(_)),
                "{body}"
            );
        }
        // Unknown elements.
        for body in [
            "<foo/>",
            "<array><bar>1</bar></array>",
            "<dict><key>a</key><x/></dict>",
        ] {
            assert_eq!(
                err_body(body).kind,
                CorruptKind::Other("unknown element"),
                "{body}"
            );
        }
        // Markup inside a text value.
        assert!(matches!(
            err_body("<string>a<b/>c</string>").kind,
            CorruptKind::Other(_)
        ));
        assert!(matches!(
            err_body("<key>a</key>").kind,
            CorruptKind::Other(_)
        ));
    }

    #[test]
    fn unbalanced_and_malformed() {
        for xml in [
            "<plist><array><string>a</string></plist>",
            "<plist><array>",
            "<plist><string>a",
            "<plist><dict><key>a</key>",
            "<plist><string>a</strin></plist>",
            "<plist><string>a</string></array></plist>",
            "<plist><string>a</string>",
            "<plist><string>a & b</string></plist>",
            "<plist><string>a &bogus; b</string></plist>",
            "<plist><string>&#0;</string></plist>",
            "<plist><string>&#xD800;</string></plist>",
        ] {
            let c = err(xml.as_bytes());
            assert!(matches!(c.kind, CorruptKind::Other(_)), "{xml}: {c}");
        }
    }

    #[test]
    fn root_rules() {
        for xml in [
            "",
            "   ",
            "<!-- only a comment -->",
            "<plist/>",
            "<plist></plist>",
            "<array/>",
            "<plist><true/><false/></plist>",
            "<plist><true/></plist><plist><true/></plist>",
            "<plist><true/></plist>trailing",
            "text<plist><true/></plist>",
        ] {
            let c = err(xml.as_bytes());
            assert!(matches!(c.kind, CorruptKind::Other(_)), "{xml:?}: {c}");
        }
        assert_eq!(parse(b"<plist><true/></plist>").unwrap(), Value::Bool(true));
    }

    #[test]
    fn external_entities_are_not_expanded() {
        let xml = "<?xml version=\"1.0\"?>\n<!DOCTYPE plist [<!ENTITY x SYSTEM \"file:///etc/passwd\">]>\n\
                   <plist><string>&x;</string></plist>";
        let c = err(xml.as_bytes());
        assert_eq!(c.kind, CorruptKind::Other("unknown entity reference"));
        let xml = "<!DOCTYPE plist [<!ENTITY y \"inline\">]><plist><string>&y;</string></plist>";
        assert_eq!(
            err(xml.as_bytes()).kind,
            CorruptKind::Other("unknown entity reference")
        );
    }

    #[test]
    fn non_utf8_text() {
        let mut xml = b"<plist><string>a".to_vec();
        xml.extend_from_slice(&[0xFF, 0xFE]);
        xml.extend_from_slice(b"</string></plist>");
        assert!(matches!(err(&xml).kind, CorruptKind::Other(_)));
        let mut xml = b"<plist><key>".to_vec();
        xml.push(0xC3);
        xml.extend_from_slice(b"</key></plist>");
        assert!(matches!(err(&xml).kind, CorruptKind::Other(_)));
    }

    #[test]
    fn locations_are_byte_offsets() {
        let xml = b"<plist><array><true/><nope/></array></plist>";
        let c = err(xml);
        assert_eq!(c.location, Location::Byte(21));
        assert_eq!(&xml[21..27], b"<nope/");
    }

    #[test]
    fn large_flat_input_is_linear() {
        // 200k elements parse quickly; a quadratic parser would not finish.
        let body = format!("<array>{}</array>", "<integer>1</integer>".repeat(200_000));
        let v = ok(&body);
        assert_eq!(v.as_array().unwrap().len(), 200_000);
        let data = format!("<data>{}</data>", "QUFB\n".repeat(200_000));
        assert_eq!(ok(&data).as_data().unwrap().len(), 600_000);
    }

    /// Valid documents for the `plist` fuzz target.
    pub fn seeds() -> Vec<(&'static str, String)> {
        vec![
            (
                "every_kind.plist",
                doc("<dict>
	<key>s</key><string>a &amp; b</string>
	<key>d</key><data>aGVsbG8=</data>
	<key>i</key><integer>-42</integer>
	<key>h</key><integer>0x1F</integer>
	<key>r</key><real>1.5</real>
	<key>t</key><true/>
	<key>f</key><false/>
	<key>date</key><date>2026-11-11T00:00:00Z</date>
	<key>a</key><array><integer>1</integer><array/><dict/></array>
</dict>"),
            ),
            (
                "sparsebundle_info.plist",
                doc("<dict>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>band-size</key>
	<integer>8388608</integer>
	<key>bundle-backingstore-version</key>
	<integer>1</integer>
	<key>diskimage-bundle-type</key>
	<string>com.apple.diskimage.sparsebundle</string>
	<key>size</key>
	<integer>1073741824</integer>
</dict>"),
            ),
            (
                "nested.plist",
                doc(&format!(
                    "{}<string>deep</string>{}",
                    "<array>".repeat(20),
                    "</array>".repeat(20)
                )),
            ),
        ]
    }

    /// Writes the seeds to `$ATLAS_FUZZ_SEEDS/plist`. Does nothing when
    /// the variable is unset. Run it with
    /// `cargo test --lib write_fuzz_seeds -- --ignored`.
    #[test]
    #[ignore = "writes files; run by the fuzz CI job"]
    fn write_fuzz_seeds() {
        let Some(root) = std::env::var_os("ATLAS_FUZZ_SEEDS") else {
            return;
        };
        let dir = std::path::Path::new(&root).join("plist");
        std::fs::create_dir_all(&dir).expect("create seed directory");
        for (name, xml) in seeds() {
            parse(xml.as_bytes()).expect("seed parses");
            std::fs::write(dir.join(name), xml).expect("write plist seed");
        }
        for (name, xml) in super::super::udif::tests::seed_plists() {
            std::fs::write(dir.join(name), xml).expect("write plist seed");
        }
    }
}
