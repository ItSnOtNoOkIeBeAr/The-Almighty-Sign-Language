// HTTP + WebSocket server on Winsock (RFC 6455, just what this protocol needs). Hand-written instead of ixwebsocket because
// the protocol needs GET /health and a 401 BEFORE the upgrade on the same port, which ixwebsocket's handshake cannot do
// (see DECISIONS.md). One reader thread and one worker thread per connection; the GPU pipeline serialises the work.
#pragma once
#include <atomic>
#include <string>

#include "config.h"
#include "processor.h"

constexpr const char* kServerVersion = "0.1.0";

class NetServer {
public:
    NetServer(const Config& cfg, IProcessor& processor, std::string token);
    ~NetServer();
    void run();   // blocks until stop()
    void stop();  // safe from a console-control handler thread
    int active_connections() const;

private:
    struct Impl;
    Impl* impl_;
};

// Protocol helpers (also unit-tested): validates a client message and returns the error code, or "" if it is a frame.
struct ParsedMessage {
    std::string error_code, error_detail;
    bool has_frame_id = false;
    unsigned frame_id = 0;
    unsigned flags = 0;
    double ts_ms = 0;
    size_t payload_offset = 0;
};
ParsedMessage parse_message(const unsigned char* data, size_t stored, size_t total_len, size_t max_bytes);

std::string sha1_base64(const std::string& in);  // Sec-WebSocket-Accept helper
