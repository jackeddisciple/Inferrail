//! ZMTP 3.0 with the NULL mechanism (rfc.zeromq.org/spec/23): enough of
//! ZeroMQ's wire protocol for PUB, SUB, ROUTER and REQ sockets to talk to
//! libzmq ones over TCP, without linking libzmq.
//!
//! Greeting as version 3.0 makes libzmq peers send subscriptions as
//! messages that start with 1 (subscribe) or 0 (cancel); the SUBSCRIBE and
//! CANCEL commands of 3.1 are understood too.

use std::io;

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

const MORE: u8 = 0x01;
const LONG: u8 = 0x02;
const COMMAND: u8 = 0x04;

/// The largest frame accepted from a peer.
pub const MAX_FRAME: u64 = 256 << 20;

/// What a peer sent: one part of a message, or a command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Frame {
    Part { more: bool, body: Vec<u8> },
    Command { name: String, data: Vec<u8> },
}

fn greeting() -> [u8; 64] {
    let mut g = [0u8; 64];
    g[0] = 0xff;
    g[9] = 0x7f;
    g[10] = 3;
    g[12..16].copy_from_slice(b"NULL");
    g
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn put_frame(out: &mut Vec<u8>, flags: u8, body: &[u8]) {
    match u8::try_from(body.len()) {
        Ok(len) => {
            out.push(flags);
            out.push(len);
        }
        Err(_) => {
            out.push(flags | LONG);
            out.extend_from_slice(&(body.len() as u64).to_be_bytes());
        }
    }
    out.extend_from_slice(body);
}

/// A command frame's bytes: its name, then its data.
pub fn command(name: &str, data: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(1 + name.len() + data.len());
    body.push(name.len() as u8);
    body.extend_from_slice(name.as_bytes());
    body.extend_from_slice(data);
    let mut out = Vec::with_capacity(body.len() + 9);
    put_frame(&mut out, COMMAND, &body);
    out
}

/// A whole message's bytes, every part but the last flagged as more.
pub fn message(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len() + 9).sum());
    for (i, part) in parts.iter().enumerate() {
        let flags = if i + 1 < parts.len() { MORE } else { 0 };
        put_frame(&mut out, flags, part);
    }
    out
}

fn ready(socket_type: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.push(11);
    data.extend_from_slice(b"Socket-Type");
    data.extend_from_slice(&(socket_type.len() as u32).to_be_bytes());
    data.extend_from_slice(socket_type.as_bytes());
    command("READY", &data)
}

/// The properties of a READY command.
fn properties(mut data: &[u8]) -> io::Result<Vec<(String, Vec<u8>)>> {
    let mut out = Vec::new();
    while !data.is_empty() {
        let name_len = usize::from(data[0]);
        let rest = &data[1..];
        if rest.len() < name_len + 4 {
            return Err(invalid("truncated READY property"));
        }
        let name = String::from_utf8_lossy(&rest[..name_len]).into_owned();
        let mut len = [0u8; 4];
        len.copy_from_slice(&rest[name_len..name_len + 4]);
        let value_len = u32::from_be_bytes(len) as usize;
        let rest = &rest[name_len + 4..];
        if rest.len() < value_len {
            return Err(invalid("truncated READY property value"));
        }
        out.push((name, rest[..value_len].to_vec()));
        data = &rest[value_len..];
    }
    Ok(out)
}

/// Exchanges greetings and READY commands as `socket_type`, and returns the
/// peer's socket type once it is one of `peers`.
pub async fn handshake<S>(stream: &mut S, socket_type: &str, peers: &[&str]) -> io::Result<String>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    stream.write_all(&greeting()).await?;
    let mut theirs = [0u8; 64];
    stream.read_exact(&mut theirs).await?;
    if theirs[0] != 0xff || theirs[9] & 0x01 != 0x01 {
        return Err(invalid("peer is not speaking ZMTP"));
    }
    if theirs[10] < 3 {
        return Err(invalid(format!("peer speaks ZMTP {}", theirs[10])));
    }
    if &theirs[12..16] != b"NULL" || theirs[16..32].iter().any(|&b| b != 0) {
        return Err(invalid("peer wants a security mechanism other than NULL"));
    }
    stream.write_all(&ready(socket_type)).await?;
    match read_frame(stream).await? {
        Frame::Command { name, data } if name == "READY" => {
            let kind = properties(&data)?
                .into_iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("Socket-Type"))
                .map(|(_, v)| String::from_utf8_lossy(&v).into_owned())
                .ok_or_else(|| invalid("READY without Socket-Type"))?;
            if !peers.contains(&kind.as_str()) {
                return Err(invalid(format!("{socket_type} cannot talk to {kind}")));
            }
            Ok(kind)
        }
        Frame::Command { name, .. } if name == "ERROR" => {
            Err(invalid("peer refused the handshake"))
        }
        _ => Err(invalid("expected READY")),
    }
}

pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Frame> {
    let flags = reader.read_u8().await?;
    let size = if flags & LONG != 0 {
        reader.read_u64().await?
    } else {
        u64::from(reader.read_u8().await?)
    };
    if size > MAX_FRAME {
        return Err(invalid(format!("frame of {size} bytes is too large")));
    }
    let mut body = vec![0u8; size as usize];
    reader.read_exact(&mut body).await?;
    if flags & COMMAND == 0 {
        return Ok(Frame::Part {
            more: flags & MORE != 0,
            body,
        });
    }
    let Some((&name_len, rest)) = body.split_first() else {
        return Err(invalid("empty command"));
    };
    let name_len = usize::from(name_len);
    if rest.len() < name_len {
        return Err(invalid("truncated command name"));
    }
    Ok(Frame::Command {
        name: String::from_utf8_lossy(&rest[..name_len]).into_owned(),
        data: rest[name_len..].to_vec(),
    })
}

/// Reads one whole message, skipping commands between messages.
pub async fn read_message<R: AsyncRead + Unpin>(reader: &mut R) -> io::Result<Vec<Vec<u8>>> {
    let mut parts = Vec::new();
    loop {
        match read_frame(reader).await? {
            Frame::Part { more, body } => {
                parts.push(body);
                if !more {
                    return Ok(parts);
                }
            }
            Frame::Command { .. } if parts.is_empty() => {}
            Frame::Command { name, .. } => {
                return Err(invalid(format!("{name} command inside a message")));
            }
        }
    }
}

/// Parses `tcp://host:port`, where `*` means every interface.
pub fn endpoint(text: &str) -> io::Result<std::net::SocketAddr> {
    use std::net::ToSocketAddrs;
    let rest = text
        .strip_prefix("tcp://")
        .ok_or_else(|| invalid(format!("{text}: only tcp:// endpoints are supported")))?;
    let rest = match rest.strip_prefix("*:") {
        Some(port) => format!("0.0.0.0:{port}"),
        None => rest.to_owned(),
    };
    rest.to_socket_addrs()?
        .next()
        .ok_or_else(|| invalid(format!("{text}: no address")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frames_pick_short_or_long_sizes() {
        assert_eq!(message(&[b"ab"]), [0, 2, b'a', b'b']);
        assert_eq!(&message(&[b"", b"x"])[..2], [MORE, 0]);
        let long = vec![7u8; 300];
        let bytes = message(&[&long]);
        assert_eq!(bytes[0], LONG);
        assert_eq!(&bytes[1..9], 300u64.to_be_bytes());
        assert_eq!(bytes.len(), 309);
    }

    #[tokio::test]
    async fn messages_and_commands_read_back() {
        let mut bytes = command("PING", b"\x00\x10ctx");
        bytes.extend(message(&[b"topic", &[0u8; 8], &vec![1u8; 1000]]));
        let mut reader = bytes.as_slice();
        let parts = read_message(&mut reader).await.unwrap();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[2].len(), 1000);
        assert!(read_frame(&mut reader).await.is_err());
        let ping = command("PING", b"ctx");
        assert_eq!(
            read_frame(&mut ping.as_slice()).await.unwrap(),
            Frame::Command {
                name: "PING".to_owned(),
                data: b"ctx".to_vec()
            }
        );
    }

    #[tokio::test]
    async fn oversized_and_truncated_frames_are_errors() {
        let mut huge = vec![LONG];
        huge.extend_from_slice(&(MAX_FRAME + 1).to_be_bytes());
        assert!(read_frame(&mut huge.as_slice()).await.is_err());
        let cut = message(&[b"hello"]);
        assert!(read_frame(&mut &cut[..4]).await.is_err());
        assert!(read_frame(&mut [COMMAND, 0].as_slice()).await.is_err());
    }

    #[tokio::test]
    async fn handshake_agrees_on_socket_types() {
        let (mut a, mut b) = tokio::io::duplex(4096);
        let (left, right) = tokio::join!(
            handshake(&mut a, "PUB", &["SUB", "XSUB"]),
            handshake(&mut b, "SUB", &["PUB", "XPUB"]),
        );
        assert_eq!(left.unwrap(), "SUB");
        assert_eq!(right.unwrap(), "PUB");
        let (mut a, mut b) = tokio::io::duplex(4096);
        let (left, _) = tokio::join!(
            handshake(&mut a, "PUB", &["SUB"]),
            handshake(&mut b, "PUB", &["SUB"]),
        );
        assert!(left.is_err());
    }

    #[tokio::test]
    async fn ready_properties_parse() {
        let Frame::Command { data, .. } =
            read_frame(&mut ready("ROUTER").as_slice()).await.unwrap()
        else {
            panic!("not a command")
        };
        assert_eq!(
            properties(&data).unwrap(),
            [("Socket-Type".to_owned(), b"ROUTER".to_vec())]
        );
        assert!(properties(&[11, b'S']).is_err());
    }

    #[test]
    fn endpoints_parse() {
        assert_eq!(endpoint("tcp://*:5557").unwrap().port(), 5557);
        assert!(endpoint("tcp://*:5557").unwrap().ip().is_unspecified());
        assert_eq!(
            endpoint("tcp://127.0.0.1:0").unwrap().to_string(),
            "127.0.0.1:0"
        );
        assert!(endpoint("ipc:///tmp/kv").is_err());
        assert!(endpoint("tcp://nohost").is_err());
    }
}
