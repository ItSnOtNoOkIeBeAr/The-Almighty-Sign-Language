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
| 6      | uint16 | `flags`        | bit0 = `end_of_stream`, bit1 = `want_keypoints` (debug view, see below) |
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
 "dropped_frames":2,"profile":"primary"}
```

- `label`: one of `labels` in `protocol/labels.json` (the instantaneous prediction; `none` when idle or unsure).
- `confidence`: smoothed probability of `label`, in `[0, 1]`.
- `state`: `idle`, `signing` or `committed`.
- `committed_label`: the word or sentence committed on this frame (one of `labels` or `sentences` in `protocol/labels.json`, e.g. `hello` or `how_are_you`), otherwise `null`. At most one event per frame. `how`, `where` and `you` may be held up to 1500 ms waiting for a partner word before they are committed (BUILD_SPEC section 8).
- `dropped_frames`: total frames on this connection that were replaced before processing (latest-only).
- `profile`: which keypoint model produced the result: `primary` (RTMW-x 384x288) or `fallback` (RTMW-l-m 256x192; the server is in safe mode or has degraded).
- `keypoints` (only when the frame had flags bit1 set): `{"w":640,"h":480,"thr":1.9,"pts":[[index,x,y,score],...]}` with x, y in pixels of the frame that was sent and `index` the COCO-WholeBody index: body 0-12 and both hands 91-132. Points scoring below `thr` are unreliable guesses; a client should not draw them.
- Servers may add extra fields. Clients must ignore unknown fields.

### Frame rate and the recognition clock

- A client may send **10 to 30 frames per second**. `client_ts_us` must be the moment the camera delivered the frame and must increase; servers resample by it.
- Keypoints (RTMW) are computed for **every** received frame, so a client that asks for `keypoints` gets a fresh skeleton in every reply (smooth overlay at any rate).
- The temporal model and the decision logic run exactly **once per 100 ms of client time** ("tick", the first tick is the first frame): each tick uses the frame whose `client_ts_us` is closest to the tick, so the 30-frame window always covers 3 s and the decision settings (M, cooldowns, `repeat_rest_frames`, ...) mean the same at 10, 20 or 30 fps. After a gap in the stream the missed ticks hold the nearest frame.
- `label`, `confidence` and `state` of a reply are the newest tick's output (so frames between ticks repeat it; `state` is `committed` only on the reply that carries a `committed_label`). `server_ms.temporal` is 0 for a frame that is not a tick. If several words are committed in one frame (after a gap) they are reported one per reply.

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

## Connection rules of the C++ server (hardening)

- The whole HTTP request must arrive within 5 s (`net.handshake_timeout_ms`), else `408` and close. Headers above 16 KB get `431`. `Sec-WebSocket-Version` must be `13` (else `426`). At most `net.max_connections` (64) connections are served; more get `503`.
- WebSocket framing is strict RFC 6455: client frames must be masked, no reserved bits, opcodes 0, 1, 2, 8, 9, 10 only, control frames at most 125 bytes and unfragmented. Any violation closes the connection with status 1002 (protocol error); a frame or message that declares more than 64 MB is refused with 1009 without being read. Messages between 2 MB and 64 MB are drained and answered with `too_large`, the connection stays open.
- Ping gets a pong with the same payload; a close frame is echoed and the connection ends.
