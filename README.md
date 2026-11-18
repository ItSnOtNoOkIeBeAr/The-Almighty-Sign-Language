# The Almighty Sign Language

Real-time sign language recognition. A laptop webcam streams JPEG frames over a Cloudflare Tunnel to a
PC with an RTX GPU. The PC recognizes signs (today with the Python fallback server; the C++/CUDA/TensorRT server follows) and sends back labels,
which the laptop shows as an overlay.

Vocabulary (ASL): hello, what, how, why, where, you (plus `none`). The sentences how_are_you and where_are_you are
joined from the words (HOW then YOU; WHERE then YOU or YOU then WHERE) within 1.5 s.

**Status:** under construction. See `governance/STATUS.md`. Full setup instructions arrive in Phase 10.

- Protocol: `protocol/PROTOCOL.md`
- Design spec: `governance/BUILD_SPEC.md`
