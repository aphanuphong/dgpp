#pragma once
// The display-reclaim span (2026-10-03): the GB10 firmware reserves 1,792 MiB
// per node for display scanout; on a headless host with nvidia_drm modeset=1
// that memory is unused but reachable — a DRM dumb buffer allocates out of
// the reservation, and once the buffer is mapped and registered with the
// CUDA driver its bytes back device accesses through a mapped device
// pointer. The mechanism was proven on these hosts by the technigmaai vLLM
// recipe (display-kv-r28.1); this is DGPP's own implementation of it
// (Apache-2.0): one dumb buffer, one mapping, one registration — sized for
// the prefix-cache arena.
//
// The span never hides a failure: init() returns false with a reason when
// the display device is unavailable (permissions, no modeset) or the driver
// refuses the registration, and the caller falls back to ordinary
// allocation. Serving never depends on the span being there.
#include <cstddef>
#include <mutex>
#include <string>

#include "common/log.hpp"

namespace dgpp {

class DisplaySpan {
 public:
  DisplaySpan() = default;
  ~DisplaySpan();
  DisplaySpan(const DisplaySpan&) = delete;
  DisplaySpan& operator=(const DisplaySpan&) = delete;

  // Allocates `bytes` of display-scanout memory (rounded up to the dumb
  // buffer's row granularity) and registers it with the CUDA runtime as
  // mapped device memory. Returns false (with a reason, no side effects)
  // when /dev/dri/card0 cannot serve the request.
  bool init(size_t bytes, std::string* error = nullptr);
  void release();

  bool active() const { return dev_ != nullptr; }
  void* base() const { return dev_; }  // the device pointer
  size_t bytes() const { return bytes_; }

 private:
  void* dev_ = nullptr;   // the registered device address
  void* map_ = nullptr;   // the host mapping (unregistered at release)
  size_t bytes_ = 0;      // the caller's request
  size_t map_bytes_ = 0;  // the dumb buffer's size (>= bytes_)
  int fd_ = -1;
  unsigned int handle_ = 0;
};

// The process-wide claim (2026-10-03): one serve process owns one prefix
// arena, so one reserved span backs it. `reserve_display_span` claims the
// span once (first caller wins, sized exactly `bytes`); the arena
// constructor's `consume_display_span` takes it, or falls back to device
// memory when nobody reserved (the knob off, or the display device
// unavailable). A consume whose size differs from the reservation is
// refused: the arena would read past the span.
//
// Header-inline so every binary that builds an arena links the registry
// without pulling display_span.cpp's DRM plumbing; the span itself lives
// in display_span.cpp.
inline std::mutex* display_span_mutex() {
  static std::mutex m;
  return &m;
}
inline void** display_span_slot() {
  static void* slot = nullptr;
  return &slot;
}
inline size_t* display_span_bytes_slot() {
  static size_t slot = 0;
  return &slot;
}

// Claims `bytes` of display memory for the process (first caller wins).
// Returns false when the display device cannot serve the span (a WARN
// names the reason); a call after a successful claim is a no-op that
// returns true.
bool reserve_display_span(size_t bytes);

// Hands the reserved span to the caller's allocation, or null when there
// is no reservation of exactly `bytes`.
inline void* consume_display_span(size_t bytes) {
  std::lock_guard<std::mutex> lock(*display_span_mutex());
  void*& slot = *display_span_slot();
  size_t& reserved = *display_span_bytes_slot();
  if (slot == nullptr) return nullptr;
  if (reserved != bytes) {
    DGPP_LOG_WARN("display span: a {}-byte arena cannot use the reserved {}-byte span", bytes,
                  reserved);
    return nullptr;
  }
  void* p = slot;
  slot = nullptr;
  reserved = 0;
  return p;
}

// The reserved-but-unconsumed span's size (0 when none or already handed
// to the arena): the serve logs use it to say what became of the backing.
inline size_t display_span_reserved_bytes() {
  std::lock_guard<std::mutex> lock(*display_span_mutex());
  return *display_span_slot() != nullptr ? *display_span_bytes_slot() : 0;
}

}  // namespace dgpp
