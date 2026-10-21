# Wire Protocol v1

This document is the contract between the laptop client, the C++ server and the
Python fallback server. Every component must follow it exactly.
`protocol/conformance_test.py` checks a running server against it.

## Endpoints

| Method | Path      | Auth | Purpose |
|--------|-----------|------|---------|
| GET    | `/health` | none | Returns `{"ok":true,"server":"cpp"\|"fallback","version":"..."}` |
| GET    | `/ws`     | Bearer | WebSocket upgrade for streaming |

### Authentication

- Header: `Authorization: Bearer <SIGNLANG_TOKEN>`.
- The server reads the expected token from the `SIGNLANG_TOKEN` environment variable only.
- If the header is missing or wrong, the server rejects with **HTTP 401 before upgrading**.

## Client to server: binary messages

All integers are **little-endian**.

| Offset | Type   | Field          | Notes |
|--------|--------|----------------|-------|
| 0      | uint32 | `magic`        | `0x534C4E47` ("SLNG") |
| 4      | uint16 | `version`      | `1` |
| 6      | uint16 | `flags`        | bit0 = `end_of_stream` |
| 8      | uint32 | `frame_id`     | monotonic per connection |
| 12     | uint64 | `client_ts_us` | client timestamp, microseconds |
| 20     | bytes  | JPEG payload   | rest of the message |

- Header size: 20 bytes.
- Maximum message size: **2 MB** (2 097 152 bytes), header included.
- A malformed message gets an `error` reply and is dropped. **The connection stays open.**
- Text messages from the client are malformed (error code `bad_type`).

## Server to client: text messages (JSON)

### Result (one per processed frame)

```json
{"type":"result","frame_id":123,"label":"hello","confidence":0.94,
 "state":"idle|signing|committed","committed_label":"hello",
 "server_ms":{"decode":1.2,"keypoints":6.1,"temporal":0.8,"total":9.0},
 "dropped_frames":2}
```

- `label`: one of `protocol/labels.json`.
- `confidence`: smoothed probability of `label`, in `[0, 1]`.
- `state`: `idle`, `signing` or `committed`.
- `committed_label`: the label committed on this frame, otherwise `null`.
- `dropped_frames`: total frames on this connection that were replaced before processing (latest-only).
- Servers may add extra fields (for example a debug `keypoints` field behind a flag). Clients must ignore unknown fields.

### Error

```json
{"type":"error","code":"bad_magic","detail":"human readable"}
```

| Code | Meaning |
|------|---------|
| `bad_type`       | text message received, binary expected |
| `too_short`      | message shorter than the 20-byte header |
| `bad_magic`      | magic field is not `0x534C4E47` |
| `bad_version`    | version field is not `1` |
| `too_large`      | message larger than 2 MB |
| `empty_payload`  | no JPEG bytes after the header |
| `decode_failed`  | the JPEG could not be decoded |
| `internal`       | server-side failure while processing; the server keeps running |

An `error` reply for a frame that has a parseable header includes `"frame_id"`.

## Flow control: latest-only

- The server keeps **one** waiting slot per connection.
- A frame arriving while another is being processed **replaces** the waiting frame. The replaced frame counts towards `dropped_frames`. Frames are never queued beyond one.
- So a client sending faster than the server can process gets results for a subset of `frame_id`s, always including the most recent one once sending stops.

## End of stream

A message with `flags` bit0 set signals the end of the stream. Its JPEG payload may be empty; if it is empty the server sends no result for it and resets the per-connection decision state.

## Conformance

```
python protocol/conformance_test.py --url ws://localhost:8765/ws --token <token>
```

Checks: auth rejection, health, valid frame round trip, malformed input, oversized input, rapid-fire frames (latest-only), and reconnect. Both servers must pass.
