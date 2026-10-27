//! vLLM v0.31.0's KV event batches in msgpack, byte for byte as msgspec
//! encodes them: a batch is an array `[ts, events, data_parallel_rank]`,
//! each event a map tagged by `type` whose fields follow in declaration
//! order, optional fields left out while they hold their default.

use std::io::{self, Read};

use rmp::Marker;
use rmp::encode::{
    ValueWriteError, write_array_len, write_f64, write_map_len, write_nil, write_sint, write_str,
    write_uint,
};

/// The storage medium vLLM names for blocks in GPU memory.
pub const MEDIUM_GPU: &str = "GPU";

#[derive(Debug, Clone, PartialEq)]
pub enum KvEvent {
    BlockStored(BlockStored),
    BlockRemoved(BlockRemoved),
    AllBlocksCleared,
}

/// Full blocks an engine has computed and cached. `block_hashes` are the
/// engine's own, in the int form vLLM publishes by default: the low 64 bits
/// of its hash.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockStored {
    pub block_hashes: Vec<u64>,
    pub parent_block_hash: Option<u64>,
    pub token_ids: Vec<u32>,
    pub block_size: u32,
    pub lora_id: Option<i64>,
    pub medium: Option<String>,
    pub lora_name: Option<String>,
    pub group_idx: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockRemoved {
    pub block_hashes: Vec<u64>,
    pub medium: Option<String>,
    pub group_idx: Option<i64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EventBatch {
    /// Seconds since the Unix epoch when the engine published the batch.
    pub ts: f64,
    pub events: Vec<KvEvent>,
    pub data_parallel_rank: Option<i64>,
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("truncated or unreadable payload: {0}")]
    Io(#[from] io::Error),
    #[error("unexpected msgpack value {found} for {field}")]
    Unexpected { field: &'static str, found: String },
    #[error("unknown event type {0:?}")]
    UnknownEvent(String),
    #[error("{0} bytes left after the batch")]
    Trailing(usize),
}

fn unexpected(field: &'static str, marker: Marker) -> DecodeError {
    DecodeError::Unexpected {
        field,
        found: format!("{marker:?}"),
    }
}

impl EventBatch {
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(64 + self.events.len() * 96);
        self.write(&mut out)
            .unwrap_or_else(|_| unreachable!("writing to a vector cannot fail"));
        out
    }

    fn write(&self, out: &mut Vec<u8>) -> Result<(), ValueWriteError> {
        let fields = if self.data_parallel_rank.is_some() {
            3
        } else {
            2
        };
        write_array_len(out, fields)?;
        write_f64(out, self.ts)?;
        write_array_len(out, len(self.events.len()))?;
        for event in &self.events {
            event.write(out)?;
        }
        if let Some(rank) = self.data_parallel_rank {
            write_int(out, rank)?;
        }
        Ok(())
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut rd = bytes;
        let fields = array_len(&mut rd, "batch")?;
        if !(2..=3).contains(&fields) {
            return Err(DecodeError::Unexpected {
                field: "batch",
                found: format!("array of {fields}"),
            });
        }
        let ts = float(&mut rd, "ts")?;
        let count = array_len(&mut rd, "events")?;
        let mut events = Vec::with_capacity(count.min(4096) as usize);
        for _ in 0..count {
            events.push(KvEvent::read(&mut rd)?);
        }
        let data_parallel_rank = if fields == 3 {
            optional_int(&mut rd, "data_parallel_rank")?
        } else {
            None
        };
        if !rd.is_empty() {
            return Err(DecodeError::Trailing(rd.len()));
        }
        Ok(Self {
            ts,
            events,
            data_parallel_rank,
        })
    }
}

fn len(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

fn write_int(out: &mut Vec<u8>, value: i64) -> Result<(), ValueWriteError> {
    match u64::try_from(value) {
        Ok(value) => write_uint(out, value).map(drop),
        Err(_) => write_sint(out, value).map(drop),
    }
}

fn write_opt_int(out: &mut Vec<u8>, value: Option<i64>) -> Result<(), ValueWriteError> {
    match value {
        Some(value) => write_int(out, value),
        None => write_nil(out).map_err(ValueWriteError::InvalidMarkerWrite),
    }
}

fn write_opt_str(out: &mut Vec<u8>, value: Option<&str>) -> Result<(), ValueWriteError> {
    match value {
        Some(value) => write_str(out, value),
        None => write_nil(out).map_err(ValueWriteError::InvalidMarkerWrite),
    }
}

fn write_hashes(out: &mut Vec<u8>, hashes: &[u64]) -> Result<(), ValueWriteError> {
    write_array_len(out, len(hashes.len()))?;
    for &hash in hashes {
        write_uint(out, hash)?;
    }
    Ok(())
}

impl KvEvent {
    fn write(&self, out: &mut Vec<u8>) -> Result<(), ValueWriteError> {
        match self {
            Self::BlockStored(e) => {
                write_map_len(out, 8 + u32::from(e.group_idx.is_some()))?;
                write_str(out, "type")?;
                write_str(out, "BlockStored")?;
                write_str(out, "block_hashes")?;
                write_hashes(out, &e.block_hashes)?;
                write_str(out, "parent_block_hash")?;
                match e.parent_block_hash {
                    Some(hash) => write_uint(out, hash).map(drop)?,
                    None => write_nil(out).map_err(ValueWriteError::InvalidMarkerWrite)?,
                }
                write_str(out, "token_ids")?;
                write_array_len(out, len(e.token_ids.len()))?;
                for &token in &e.token_ids {
                    write_uint(out, u64::from(token))?;
                }
                write_str(out, "block_size")?;
                write_uint(out, u64::from(e.block_size))?;
                write_str(out, "lora_id")?;
                write_opt_int(out, e.lora_id)?;
                write_str(out, "medium")?;
                write_opt_str(out, e.medium.as_deref())?;
                write_str(out, "lora_name")?;
                write_opt_str(out, e.lora_name.as_deref())?;
                if let Some(group) = e.group_idx {
                    write_str(out, "group_idx")?;
                    write_int(out, group)?;
                }
            }
            Self::BlockRemoved(e) => {
                write_map_len(out, 3 + u32::from(e.group_idx.is_some()))?;
                write_str(out, "type")?;
                write_str(out, "BlockRemoved")?;
                write_str(out, "block_hashes")?;
                write_hashes(out, &e.block_hashes)?;
                write_str(out, "medium")?;
                write_opt_str(out, e.medium.as_deref())?;
                if let Some(group) = e.group_idx {
                    write_str(out, "group_idx")?;
                    write_int(out, group)?;
                }
            }
            Self::AllBlocksCleared => {
                write_map_len(out, 1)?;
                write_str(out, "type")?;
                write_str(out, "AllBlocksCleared")?;
            }
        }
        Ok(())
    }

    /// Reads one event. Fields this version does not know are skipped, so
    /// newer engines that add some still decode.
    fn read(rd: &mut &[u8]) -> Result<Self, DecodeError> {
        let fields = map_len(rd, "event")?;
        let mut kind: Option<String> = None;
        let mut hashes = Vec::new();
        let mut parent = None;
        let mut tokens = Vec::new();
        let mut block_size = 0;
        let mut lora_id = None;
        let mut medium = None;
        let mut lora_name = None;
        let mut group_idx = None;
        for _ in 0..fields {
            let key = string(rd, "key")?;
            match key.as_str() {
                "type" => kind = Some(string(rd, "type")?),
                "block_hashes" => {
                    let n = array_len(rd, "block_hashes")?;
                    hashes = (0..n)
                        .map(|_| hash(rd, "block_hashes"))
                        .collect::<Result<_, _>>()?;
                }
                "parent_block_hash" => {
                    parent = if peek_nil(rd) {
                        skip(rd)?;
                        None
                    } else {
                        Some(hash(rd, "parent_block_hash")?)
                    };
                }
                "token_ids" => {
                    let n = array_len(rd, "token_ids")?;
                    tokens = (0..n)
                        .map(|_| {
                            let id = uint(rd, "token_ids")?;
                            u32::try_from(id).map_err(|_| DecodeError::Unexpected {
                                field: "token_ids",
                                found: id.to_string(),
                            })
                        })
                        .collect::<Result<_, _>>()?;
                }
                "block_size" => {
                    block_size = u32::try_from(uint(rd, "block_size")?).unwrap_or(u32::MAX);
                }
                "lora_id" => lora_id = optional_int(rd, "lora_id")?,
                "medium" => medium = optional_string(rd, "medium")?,
                "lora_name" => lora_name = optional_string(rd, "lora_name")?,
                "group_idx" => group_idx = optional_int(rd, "group_idx")?,
                _ => skip(rd)?,
            }
        }
        match kind.as_deref() {
            Some("BlockStored") => Ok(Self::BlockStored(BlockStored {
                block_hashes: hashes,
                parent_block_hash: parent,
                token_ids: tokens,
                block_size,
                lora_id,
                medium,
                lora_name,
                group_idx,
            })),
            Some("BlockRemoved") => Ok(Self::BlockRemoved(BlockRemoved {
                block_hashes: hashes,
                medium,
                group_idx,
            })),
            Some("AllBlocksCleared") => Ok(Self::AllBlocksCleared),
            other => Err(DecodeError::UnknownEvent(other.unwrap_or("").to_owned())),
        }
    }
}

fn marker(rd: &mut &[u8]) -> Result<Marker, DecodeError> {
    rmp::decode::read_marker(rd).map_err(|e| DecodeError::Io(e.0))
}

fn take(rd: &mut &[u8], n: usize) -> Result<Vec<u8>, DecodeError> {
    let mut out = vec![0; n];
    rd.read_exact(&mut out)?;
    Ok(out)
}

fn be(rd: &mut &[u8], n: usize) -> Result<u64, DecodeError> {
    Ok(take(rd, n)?
        .iter()
        .fold(0u64, |acc, b| acc << 8 | u64::from(*b)))
}

fn peek_nil(rd: &[u8]) -> bool {
    rd.first() == Some(&0xc0)
}

fn array_len(rd: &mut &[u8], field: &'static str) -> Result<u32, DecodeError> {
    match marker(rd)? {
        Marker::FixArray(n) => Ok(u32::from(n)),
        Marker::Array16 => Ok(be(rd, 2)? as u32),
        Marker::Array32 => Ok(be(rd, 4)? as u32),
        other => Err(unexpected(field, other)),
    }
}

fn map_len(rd: &mut &[u8], field: &'static str) -> Result<u32, DecodeError> {
    match marker(rd)? {
        Marker::FixMap(n) => Ok(u32::from(n)),
        Marker::Map16 => Ok(be(rd, 2)? as u32),
        Marker::Map32 => Ok(be(rd, 4)? as u32),
        other => Err(unexpected(field, other)),
    }
}

fn int(rd: &mut &[u8], field: &'static str) -> Result<i128, DecodeError> {
    Ok(match marker(rd)? {
        Marker::FixPos(n) => i128::from(n),
        Marker::FixNeg(n) => i128::from(n),
        Marker::U8 => i128::from(be(rd, 1)?),
        Marker::U16 => i128::from(be(rd, 2)?),
        Marker::U32 => i128::from(be(rd, 4)?),
        Marker::U64 => i128::from(be(rd, 8)?),
        Marker::I8 => i128::from(be(rd, 1)? as u8 as i8),
        Marker::I16 => i128::from(be(rd, 2)? as u16 as i16),
        Marker::I32 => i128::from(be(rd, 4)? as u32 as i32),
        Marker::I64 => i128::from(be(rd, 8)? as i64),
        other => return Err(unexpected(field, other)),
    })
}

fn uint(rd: &mut &[u8], field: &'static str) -> Result<u64, DecodeError> {
    let value = int(rd, field)?;
    u64::try_from(value).map_err(|_| DecodeError::Unexpected {
        field,
        found: value.to_string(),
    })
}

fn optional_int(rd: &mut &[u8], field: &'static str) -> Result<Option<i64>, DecodeError> {
    if peek_nil(rd) {
        skip(rd)?;
        return Ok(None);
    }
    let value = int(rd, field)?;
    i64::try_from(value)
        .map(Some)
        .map_err(|_| DecodeError::Unexpected {
            field,
            found: value.to_string(),
        })
}

/// A block hash in either form vLLM publishes: an int, or the hash bytes,
/// whose low 64 bits are the int form.
fn hash(rd: &mut &[u8], field: &'static str) -> Result<u64, DecodeError> {
    let n = match rd.first() {
        Some(0xc4) => {
            marker(rd)?;
            be(rd, 1)? as usize
        }
        Some(0xc5) => {
            marker(rd)?;
            be(rd, 2)? as usize
        }
        Some(0xc6) => {
            marker(rd)?;
            be(rd, 4)? as usize
        }
        _ => return uint(rd, field),
    };
    let bytes = take(rd, n)?;
    Ok(bytes
        .iter()
        .rev()
        .take(8)
        .rev()
        .fold(0u64, |acc, b| acc << 8 | u64::from(*b)))
}

fn float(rd: &mut &[u8], field: &'static str) -> Result<f64, DecodeError> {
    match rd.first() {
        Some(0xcb) => {
            marker(rd)?;
            Ok(f64::from_bits(be(rd, 8)?))
        }
        Some(0xca) => {
            marker(rd)?;
            Ok(f64::from(f32::from_bits(be(rd, 4)? as u32)))
        }
        _ => Ok(int(rd, field)? as f64),
    }
}

fn string(rd: &mut &[u8], field: &'static str) -> Result<String, DecodeError> {
    let n = match marker(rd)? {
        Marker::FixStr(n) => usize::from(n),
        Marker::Str8 => be(rd, 1)? as usize,
        Marker::Str16 => be(rd, 2)? as usize,
        Marker::Str32 => be(rd, 4)? as usize,
        other => return Err(unexpected(field, other)),
    };
    String::from_utf8(take(rd, n)?).map_err(|_| DecodeError::Unexpected {
        field,
        found: "invalid UTF-8".to_owned(),
    })
}

fn optional_string(rd: &mut &[u8], field: &'static str) -> Result<Option<String>, DecodeError> {
    if peek_nil(rd) {
        skip(rd)?;
        return Ok(None);
    }
    string(rd, field).map(Some)
}

/// Skips one value of any type, however deeply nested.
fn skip(rd: &mut &[u8]) -> Result<(), DecodeError> {
    let mut pending: u64 = 1;
    while pending > 0 {
        pending -= 1;
        let bytes = match marker(rd)? {
            Marker::Null | Marker::True | Marker::False => 0,
            Marker::FixPos(_) | Marker::FixNeg(_) => 0,
            Marker::U8 | Marker::I8 => 1,
            Marker::U16 | Marker::I16 => 2,
            Marker::U32 | Marker::I32 | Marker::F32 => 4,
            Marker::U64 | Marker::I64 | Marker::F64 => 8,
            Marker::FixStr(n) => usize::from(n),
            Marker::Str8 | Marker::Bin8 => be(rd, 1)? as usize,
            Marker::Str16 | Marker::Bin16 => be(rd, 2)? as usize,
            Marker::Str32 | Marker::Bin32 => be(rd, 4)? as usize,
            Marker::FixArray(n) => {
                pending += u64::from(n);
                0
            }
            Marker::Array16 => {
                pending += be(rd, 2)?;
                0
            }
            Marker::Array32 => {
                pending += be(rd, 4)?;
                0
            }
            Marker::FixMap(n) => {
                pending += 2 * u64::from(n);
                0
            }
            Marker::Map16 => {
                pending += 2 * be(rd, 2)?;
                0
            }
            Marker::Map32 => {
                pending += 2 * be(rd, 4)?;
                0
            }
            Marker::FixExt1 => 2,
            Marker::FixExt2 => 3,
            Marker::FixExt4 => 5,
            Marker::FixExt8 => 9,
            Marker::FixExt16 => 17,
            Marker::Ext8 => be(rd, 1)? as usize + 1,
            Marker::Ext16 => be(rd, 2)? as usize + 1,
            Marker::Ext32 => be(rd, 4)? as usize + 1,
            Marker::Reserved => return Err(unexpected("value", Marker::Reserved)),
        };
        if bytes > rd.len() {
            return Err(DecodeError::Io(io::ErrorKind::UnexpectedEof.into()));
        }
        *rd = &rd[bytes..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn stored(hashes: Vec<u64>, parent: Option<u64>, tokens: Vec<u32>) -> KvEvent {
        KvEvent::BlockStored(BlockStored {
            block_hashes: hashes,
            parent_block_hash: parent,
            token_ids: tokens,
            block_size: 16,
            lora_id: None,
            medium: Some(MEDIUM_GPU.to_owned()),
            lora_name: None,
            group_idx: Some(0),
        })
    }

    fn removed(hash: u64) -> KvEvent {
        KvEvent::BlockRemoved(BlockRemoved {
            block_hashes: vec![hash],
            medium: Some(MEDIUM_GPU.to_owned()),
            group_idx: Some(0),
        })
    }

    /// The batches `make_fixture.py` encodes with vLLM's structs.
    fn fixture_batches() -> Vec<EventBatch> {
        vec![
            EventBatch {
                ts: 1793000000.25,
                events: vec![stored(
                    vec![0x1234, 0xFEDC_BA98_7654_3210],
                    None,
                    (0..32).collect(),
                )],
                data_parallel_rank: Some(0),
            },
            EventBatch {
                ts: 1793000000.5,
                events: vec![
                    stored(
                        vec![7],
                        Some(0xFEDC_BA98_7654_3210),
                        (0..16).map(|i| 128_000 + i).collect(),
                    ),
                    removed(0x1234),
                    removed(u64::MAX),
                ],
                data_parallel_rank: Some(0),
            },
            EventBatch {
                ts: 1793000001.0,
                events: vec![KvEvent::AllBlocksCleared],
                data_parallel_rank: Some(0),
            },
        ]
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn batches_encode_byte_for_byte_as_vllm_does() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/vllm/kv_events/batches.jsonl"
        );
        let text = std::fs::read_to_string(path).expect("fixture");
        let lines: Vec<serde_json::Value> = text
            .lines()
            .map(|l| serde_json::from_str(l).expect("json line"))
            .collect();
        let batches = fixture_batches();
        assert_eq!(lines.len(), batches.len());
        for (line, batch) in lines.iter().zip(&batches) {
            let want = line["payload"].as_str().expect("payload");
            assert_eq!(hex(&batch.encode()), want);
            let bytes: Vec<u8> = (0..want.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&want[i..i + 2], 16).expect("hex"))
                .collect();
            assert_eq!(&EventBatch::decode(&bytes).expect("decodes"), batch);
        }
    }

    #[test]
    fn unknown_fields_are_skipped_and_hash_bytes_read_as_ints() {
        let mut out = Vec::new();
        write_array_len(&mut out, 2).unwrap();
        write_f64(&mut out, 1.5).unwrap();
        write_array_len(&mut out, 1).unwrap();
        write_map_len(&mut out, 5).unwrap();
        write_str(&mut out, "type").unwrap();
        write_str(&mut out, "BlockRemoved").unwrap();
        write_str(&mut out, "block_hashes").unwrap();
        write_array_len(&mut out, 1).unwrap();
        rmp::encode::write_bin(&mut out, &[0xaa; 24]).unwrap();
        write_str(&mut out, "medium").unwrap();
        write_nil(&mut out).unwrap();
        write_str(&mut out, "locality").unwrap();
        write_array_len(&mut out, 2).unwrap();
        write_map_len(&mut out, 1).unwrap();
        write_str(&mut out, "x").unwrap();
        write_sint(&mut out, -5).unwrap();
        write_str(&mut out, "deep").unwrap();
        write_str(&mut out, "session_id").unwrap();
        write_str(&mut out, "s").unwrap();
        let batch = EventBatch::decode(&out).unwrap();
        assert_eq!(batch.data_parallel_rank, None);
        assert_eq!(
            batch.events,
            [KvEvent::BlockRemoved(BlockRemoved {
                block_hashes: vec![0xaaaa_aaaa_aaaa_aaaa],
                medium: None,
                group_idx: None,
            })]
        );
    }

    #[test]
    fn malformed_payloads_are_errors_not_panics() {
        let good = fixture_batches()[1].encode();
        for cut in 0..good.len() {
            assert!(EventBatch::decode(&good[..cut]).is_err(), "cut {cut}");
        }
        let mut long = good.clone();
        long.push(0);
        assert!(matches!(
            EventBatch::decode(&long),
            Err(DecodeError::Trailing(1))
        ));
        let mut out = Vec::new();
        write_array_len(&mut out, 2).unwrap();
        write_f64(&mut out, 1.0).unwrap();
        write_array_len(&mut out, 1).unwrap();
        write_map_len(&mut out, 1).unwrap();
        write_str(&mut out, "type").unwrap();
        write_str(&mut out, "BlockMoved").unwrap();
        assert!(matches!(
            EventBatch::decode(&out),
            Err(DecodeError::UnknownEvent(_))
        ));
    }

    fn event() -> impl Strategy<Value = KvEvent> {
        let opt_str = prop::option::of("[a-zA-Z0-9_]{0,40}");
        prop_oneof![
            (
                prop::collection::vec(any::<u64>(), 0..20),
                prop::option::of(any::<u64>()),
                prop::collection::vec(any::<u32>(), 0..300),
                any::<u32>(),
                prop::option::of(any::<i64>()),
                opt_str.clone(),
                opt_str,
                prop::option::of(any::<i64>()),
            )
                .prop_map(
                    |(
                        block_hashes,
                        parent_block_hash,
                        token_ids,
                        block_size,
                        lora_id,
                        medium,
                        lora_name,
                        group_idx,
                    )| {
                        KvEvent::BlockStored(BlockStored {
                            block_hashes,
                            parent_block_hash,
                            token_ids,
                            block_size,
                            lora_id,
                            medium,
                            lora_name,
                            group_idx,
                        })
                    }
                ),
            (
                prop::collection::vec(any::<u64>(), 0..20),
                prop::option::of("[A-Z]{0,8}"),
                prop::option::of(any::<i64>()),
            )
                .prop_map(|(block_hashes, medium, group_idx)| {
                    KvEvent::BlockRemoved(BlockRemoved {
                        block_hashes,
                        medium,
                        group_idx,
                    })
                }),
            Just(KvEvent::AllBlocksCleared),
        ]
    }

    proptest! {
        #[test]
        fn every_batch_round_trips(
            ts in any::<f64>().prop_filter("finite", |t| t.is_finite()),
            events in prop::collection::vec(event(), 0..12),
            data_parallel_rank in prop::option::of(any::<i64>()),
        ) {
            let batch = EventBatch { ts, events, data_parallel_rank };
            prop_assert_eq!(EventBatch::decode(&batch.encode()).unwrap(), batch);
        }

        #[test]
        fn random_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..400)) {
            let _ = EventBatch::decode(&bytes);
        }
    }
}
