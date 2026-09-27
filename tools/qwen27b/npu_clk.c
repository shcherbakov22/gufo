// Reads the NPU's *granted* clocks and TOPS from the amdxdna DRM query
// interface. aie2_update_counters() refreshes these from the SMU metrics table
// (npu4_regs.c), so unlike the cached request in the driver these are what the
// SMU actually granted.
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include <drm/amdxdna_accel.h>
#include <drm/drm.h>

static int get_clocks(int fd) {
  struct amdxdna_drm_query_clock_metadata clk;
  struct amdxdna_drm_get_info info;
  memset(&clk, 0, sizeof(clk));
  info.param = DRM_AMDXDNA_QUERY_CLOCK_METADATA;
  info.buffer_size = sizeof(clk);
  info.buffer = (unsigned long)&clk;
  if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info) < 0) return -1;
  printf("mp_npu=%.*s:%u h=%.*s:%u", 16, clk.mp_npu_clock.name,
         clk.mp_npu_clock.freq_mhz, 16, clk.h_clock.name,
         clk.h_clock.freq_mhz);
  return 0;
}

static int get_res(int fd) {
  struct amdxdna_drm_get_resource_info res;
  struct amdxdna_drm_get_info info;
  memset(&res, 0, sizeof(res));
  info.param = DRM_AMDXDNA_QUERY_RESOURCE_INFO;
  info.buffer_size = sizeof(res);
  info.buffer = (unsigned long)&res;
  if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info) < 0) return -1;
  printf(" clk_max=%llu tops_curr=%llu/%llu", (unsigned long long)res.npu_clk_max,
         (unsigned long long)res.npu_tops_curr,
         (unsigned long long)res.npu_tops_max);
  return 0;
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "/dev/accel/accel0";
  int interval_ms = argc > 2 ? atoi(argv[2]) : 0;
  int count = argc > 3 ? atoi(argv[3]) : 1;
  int fd = open(path, O_RDWR);
  if (fd < 0) {
    fprintf(stderr, "open %s: %s\n", path, strerror(errno));
    return 1;
  }
  for (int i = 0; i < count; ++i) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("%ld.%03ld ", (long)ts.tv_sec, ts.tv_nsec / 1000000);
    if (get_clocks(fd) || get_res(fd)) {
      fprintf(stderr, "query failed: %s\n", strerror(errno));
      return 1;
    }
    printf("\n");
    fflush(stdout);
    if (i + 1 < count && interval_ms > 0) usleep(interval_ms * 1000);
  }
  close(fd);
  return 0;
}
