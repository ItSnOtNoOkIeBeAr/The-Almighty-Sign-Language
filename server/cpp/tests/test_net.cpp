// Protocol helpers: handshake hash and the client-message validation (mirrors fallback_server.parse).
#include <gtest/gtest.h>

#include <cstring>
#include <vector>

#include "net.h"

TEST(Net, Rfc6455HandshakeExample) {
    EXPECT_EQ(sha1_base64("dGhlIHNhbXBsZSBub25jZQ==258EAFA5-E914-47DA-95CA-C5AB0DC85B11"), "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=");
}

namespace {
std::vector<unsigned char> msg(uint32_t magic, uint16_t ver, uint16_t flags, uint32_t fid, size_t payload) {
    std::vector<unsigned char> m(20 + payload, 0xAB);
    const uint64_t ts = 123456;
    std::memcpy(&m[0], &magic, 4), std::memcpy(&m[4], &ver, 2), std::memcpy(&m[6], &flags, 2);
    std::memcpy(&m[8], &fid, 4), std::memcpy(&m[12], &ts, 8);
    return m;
}
constexpr size_t MAXB = 2 * 1024 * 1024;
}  // namespace

TEST(Net, ValidFrame) {
    const auto m = msg(0x534C4E47, 1, 0, 9, 100);
    const auto p = parse_message(m.data(), m.size(), m.size(), MAXB);
    EXPECT_TRUE(p.error_code.empty());
    EXPECT_EQ(p.frame_id, 9u);
    EXPECT_EQ(p.payload_offset, 20u);
    EXPECT_DOUBLE_EQ(p.ts_ms, 123.456);
}

TEST(Net, ErrorCodes) {
    auto m = msg(0x534C4E47, 1, 0, 3, 10);
    EXPECT_EQ(parse_message(m.data(), 4, 4, MAXB).error_code, "too_short");
    m = msg(0xDEADBEEF, 1, 0, 3, 10);
    auto p = parse_message(m.data(), m.size(), m.size(), MAXB);
    EXPECT_EQ(p.error_code, "bad_magic");
    EXPECT_FALSE(p.has_frame_id);
    m = msg(0x534C4E47, 99, 0, 3, 10);
    p = parse_message(m.data(), m.size(), m.size(), MAXB);
    EXPECT_EQ(p.error_code, "bad_version");
    EXPECT_TRUE(p.has_frame_id);
    m = msg(0x534C4E47, 1, 0, 3, 0);
    EXPECT_EQ(parse_message(m.data(), m.size(), m.size(), MAXB).error_code, "empty_payload");
    m = msg(0x534C4E47, 1, 1, 3, 0);  // empty end_of_stream is valid
    EXPECT_TRUE(parse_message(m.data(), m.size(), m.size(), MAXB).error_code.empty());
    m = msg(0x534C4E47, 1, 0, 3, 50);
    p = parse_message(m.data(), m.size(), MAXB + 1, MAXB);  // only the prefix was stored, the full length is what counts
    EXPECT_EQ(p.error_code, "too_large");
    EXPECT_EQ(p.frame_id, 3u);
}
