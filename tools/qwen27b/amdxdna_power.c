// amdxdna_power.c -- read NPU telemetry through the amdxdna GET_INFO ioctl.
//
// ai2_get_sensors() builds AMDXDNA_SENSOR_TYPE_POWER from amd_pmf_npu_metrics,
// which the PMF driver derives from the SPS telemetry table.  That is the
// platform/SPS layer, the only layer where the NPU has its own rail
// (VDDCR_NPU) and its own TDC throttler.  Read-only.
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>

#include "amdxdna_accel.h"

#define MAXS 16

int main(int argc, char **argv) {
  const int n = argc > 1 ? atoi(argv[1]) : 1;
  const int gap_ms = argc > 2 ? atoi(argv[2]) : 0;
  const char *dev = argc > 3 ? argv[3] : "/dev/accel/accel0";

  int fd = open(dev, O_RDWR);
  if (fd < 0) { perror("open"); return 1; }

  struct amdxdna_drm_query_sensor *buf = calloc(MAXS, sizeof(*buf));
  if (!buf) return 1;

  for (int k = 0; k < n; k++) {
    memset(buf, 0, MAXS * sizeof(*buf));
    struct amdxdna_drm_get_info info = {0};
    info.param = DRM_AMDXDNA_QUERY_SENSORS;
    info.buffer_size = MAXS * sizeof(*buf);
    info.buffer = (uint64_t)(uintptr_t)buf;
    if (ioctl(fd, DRM_IOCTL_AMDXDNA_GET_INFO, &info)) { perror("ioctl"); return 2; }
    const int cnt = (int)(info.buffer_size / sizeof(*buf));

    printf("t=%ld sensors=%d", (long)time(NULL), cnt);
    for (int i = 0; i < cnt; i++)
      if (buf[i].type == AMDXDNA_SENSOR_TYPE_POWER)
        printf("  POWER=%u%s", buf[i].input, buf[i].units);
    printf("  cols:");
    for (int i = 0; i < cnt; i++)
      if (buf[i].type == AMDXDNA_SENSOR_TYPE_COLUMN_UTILIZATION)
        printf(" %u", buf[i].input);
    printf("\n");
    fflush(stdout);
    if (gap_ms) usleep(gap_ms * 1000);
  }
  close(fd);
  return 0;
}