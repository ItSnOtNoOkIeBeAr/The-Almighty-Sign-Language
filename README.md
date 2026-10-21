# The Almighty Sign Language

Real-time sign language recognition. A laptop webcam streams JPEG frames over a Cloudflare Tunnel to a
PC with an RTX GPU. The PC recognizes signs with a C++/CUDA/TensorRT pipeline and sends back labels,
which the laptop shows as an overlay.

Vocabulary: idle, hello, hi, what, how, why, how_are_you, where_are_you.

**Status:** under construction. See `STATUS.md`. Full setup instructions arrive in Phase 10.

- Protocol: `protocol/PROTOCOL.md`
- Design spec: `BUILD_SPEC.md`
