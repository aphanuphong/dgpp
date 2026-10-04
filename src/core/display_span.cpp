#include "core/display_span.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda_runtime.h>

#include "common/log.hpp"

namespace dgpp {

// The DRM UAPI definitions are inlined here rather than included: the
// cross-build's ARM64 sysroot carries no DRM headers (checked 2026-10-03),
// and the ioctls are a stable kernel ABI. The structs match
// <drm/drm_mode.h> byte for byte; the ioctl numbers use the system's
// _IOWR encoding so they are right for the target's assembly.
namespace drm_uapi {

struct CreateDumb {
  unsigned int height;
  unsigned int width;
  unsigned int bpp;
  unsigned int flags;
  unsigned int handle;
  unsigned int pitch;
  unsigned long long size;
};

struct MapDumb {
  unsigned int handle;
  unsigned int pad;
  unsigned long long offset;
};

struct DestroyDumb {
  unsigned int handle;
};

// <sys/ioctl.h>'s _IOWR: direction | size | type | nr.
// Verified against the kernel UAPI on GB10 (driver 580.178.04):
//   DRM_IOCTL_MODE_CREATE_DUMB  = 0xc02064b2
//   DRM_IOCTL_MODE_MAP_DUMB     = 0xc01064b3
//   DRM_IOCTL_MODE_DESTROY_DUMB = 0xc00464b4
// i.e. type 'd', nr 0xB2..0xB4, direction _IOWR (3 << 30). The old
// _IOW encoding (2 << 30) made the kernel reject every call with
// -EINVAL because the command never matched the dispatch switch.
constexpr unsigned int kIoctlType = 'd';
constexpr unsigned int kNrCreate = 0xB2;
constexpr unsigned int kNrMap = 0xB3;
constexpr unsigned int kNrDestroy = 0xB4;

template <class T>
constexpr unsigned int iowr(unsigned int nr) {
  return (3U << 30) | ((sizeof(T) & 0x3FFFU) << 16) | (kIoctlType << 8) | nr;
}

constexpr unsigned int kIoctlCreate = iowr<CreateDumb>(kNrCreate);
constexpr unsigned int kIoctlMap = iowr<MapDumb>(kNrMap);
constexpr unsigned int kIoctlDestroy = iowr<DestroyDumb>(kNrDestroy);

}  // namespace drm_uapi

namespace {

constexpr unsigned int kDumbWidth = 4096;
constexpr unsigned int kDumbBpp = 32;

std::string errno_text(const char* what) {
  return std::string(what) + ": " + std::strerror(errno);
}

}  // namespace

DisplaySpan::~DisplaySpan() {
  release();
}

bool DisplaySpan::init(size_t bytes, std::string* error) {
  auto fail = [&](const std::string& why) {
    if (error) *error = why;
    DGPP_LOG_DEBUG("display span: {}", why);
    return false;
  };
  if (bytes == 0) return fail("display span: zero-byte request");
  if (dev_) return fail("display span: already initialized");

  fd_ = open("/dev/dri/card0", O_RDWR | O_CLOEXEC);
  if (fd_ < 0)
    return fail(
        errno_text("display span: open(/dev/dri/card0) failed (is the "
                   "user in the video group, is nvidia_drm modeset=1?)"));

  drm_uapi::CreateDumb creq{};
  creq.width = kDumbWidth;
  creq.bpp = kDumbBpp;
  const size_t row = static_cast<size_t>(kDumbWidth) * (kDumbBpp / 8);
  creq.height = static_cast<unsigned int>((bytes + row - 1) / row);
  if (ioctl(fd_, drm_uapi::kIoctlCreate, &creq) != 0) {
    const std::string why = errno_text("display span: CREATE_DUMB failed");
    release();
    return fail(why);
  }
  if (creq.size < bytes || creq.pitch == 0) {
    release();
    return fail("display span: the driver returned a buffer smaller than the request (" +
                std::to_string(creq.size) + " < " + std::to_string(bytes) + ")");
  }
  handle_ = creq.handle;
  map_bytes_ = static_cast<size_t>(creq.size);

  drm_uapi::MapDumb mreq{};
  mreq.handle = handle_;
  if (ioctl(fd_, drm_uapi::kIoctlMap, &mreq) != 0) {
    const std::string why = errno_text("display span: MAP_DUMB failed");
    release();
    return fail(why);
  }
  map_ = mmap(nullptr, map_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_,
              static_cast<off_t>(mreq.offset));
  if (map_ == MAP_FAILED) {
    map_ = nullptr;
    const std::string why = errno_text("display span: mmap of the dumb buffer failed");
    release();
    return fail(why);
  }

  // The display carveout is I/O memory to the driver: register it with the
  // mapped + io-memory flags so CUDA hands out a device address for it.
  cudaError_t e =
      cudaHostRegister(map_, map_bytes_, cudaHostRegisterMapped | cudaHostRegisterIoMemory);
  if (e != cudaSuccess) {
    const std::string why =
        std::string("display span: cudaHostRegister failed: ") + cudaGetErrorString(e);
    release();
    return fail(why);
  }
  dev_ = nullptr;
  e = cudaHostGetDevicePointer(&dev_, map_, 0);
  if (e != cudaSuccess || dev_ == nullptr) {
    const std::string why =
        std::string("display span: cudaHostGetDevicePointer failed: ") + cudaGetErrorString(e);
    release();
    return fail(why);
  }
  bytes_ = bytes;
  DGPP_LOG_INFO(
      "display span: {} MiB of display-reclaimed memory at {} (dumb buffer "
      "{} x {} x {}bpp, {} KiB rows)",
      bytes_ >> 20, dev_, kDumbWidth, creq.height, kDumbBpp, creq.pitch >> 10);
  return true;
}

void DisplaySpan::release() {
  if (map_) {
    (void)cudaHostUnregister(map_);
    munmap(map_, map_bytes_);
    map_ = nullptr;
  }
  if (fd_ >= 0) {
    if (handle_ != 0) {
      drm_uapi::DestroyDumb dreq{};
      dreq.handle = handle_;
      (void)ioctl(fd_, drm_uapi::kIoctlDestroy, &dreq);
      handle_ = 0;
    }
    close(fd_);
    fd_ = -1;
  }
  dev_ = nullptr;
  bytes_ = map_bytes_ = 0;
}

bool reserve_display_span(size_t bytes) {
  std::lock_guard<std::mutex> lock(*display_span_mutex());
  void*& slot = *display_span_slot();
  size_t& reserved = *display_span_bytes_slot();
  if (slot != nullptr) {
    if (reserved == bytes) return true;  // already claimed at this size
    DGPP_LOG_WARN("display span: refusing a {}-byte claim: {} bytes already reserved", bytes,
                  reserved);
    return false;
  }
  static DisplaySpan span;  // the process's one claim; freed at exit
  std::string error;
  if (!span.init(bytes, &error)) {
    DGPP_LOG_WARN("{}", error);
    return false;
  }
  slot = span.base();
  reserved = bytes;
  return true;
}

}  // namespace dgpp
