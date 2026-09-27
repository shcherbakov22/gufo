// Talk to the NPU's SMU mailbox from userspace via BAR5.
// Mirrors aie_smu_exec(): each register is a 32-bit word in the NPU's BAR5,
// with offsets taken from npu4_regs.c (NPU4_SMU_BAR_BASE = MMNPU_APERTURE4_BASE).
//   SMU_CMD  = MP1_C2PMSG_0  0x3B10900 -> +0x900
//   SMU_ARG  = MP1_C2PMSG_60 0x3B109F0 -> +0x9F0   (also the OUT register)
//   SMU_RESP = MP1_C2PMSG_61 0x3B109F4 -> +0x9F4
//   SMU_INTR = APERTURE4     0x3B10000 -> +0x000
// Messages (aie_smu.c): 3 POWER_ON, 4 POWER_OFF, 5 SET_MPNPUCLK_FREQ,
//                       6 SET_HCLK_FREQ, 7 SET_SOFT_DPMLEVEL, 8 SET_HARD_DPMLEVEL
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>

#define OFF_CMD  0x900
#define OFF_ARG  0x9F0
#define OFF_RESP 0x9F4
#define OFF_INTR 0x000
#define BAR_SIZE 4096

static volatile uint8_t *bar;
static inline uint32_t rd(unsigned o) { return *(volatile uint32_t *)(bar + o); }
static inline void wr(unsigned o, uint32_t v) { *(volatile uint32_t *)(bar + o) = v; }

static int smu_exec(uint32_t cmd, uint32_t arg, uint32_t *out, int timeout_ms) {
  struct timespec ts = {0, 1000000};
  int i, n = timeout_ms;
  wr(OFF_RESP, 0);
  wr(OFF_ARG, arg);
  wr(OFF_CMD, cmd);
  wr(OFF_INTR, 0);
  wr(OFF_INTR, 1);
  for (i = 0; i < n; i++) {
    uint32_t r = rd(OFF_RESP);
    if (r) {
      if (out) *out = rd(OFF_ARG);
      return (r == 1) ? 0 : (int)r;   /* SMU_RESULT_OK == 1 */
    }
    nanosleep(&ts, NULL);
  }
  return -1;
}

int main(int argc, char **argv) {
  const char *path = argc > 3 ? argv[3] : "/sys/bus/pci/devices/0000:c5:00.1/resource5";
  int fd, ret, peek = (argc > 4);
  uint32_t cmd, arg = 0, out = 0;
  void *m;

  if (argc < 2) { fprintf(stderr, "usage: %s <cmd> [arg] [resource_path] [peek]\n", argv[0]); return 2; }
  cmd = strtoul(argv[1], NULL, 0);
  if (argc > 2) arg = strtoul(argv[2], NULL, 0);

  fd = open(path, O_RDWR | O_SYNC);
  if (fd < 0) { fprintf(stderr, "open %s: %s\n", path, strerror(errno)); return 1; }
  m = mmap(NULL, BAR_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  if (m == MAP_FAILED) { fprintf(stderr, "mmap %s: %s\n", path, strerror(errno)); return 1; }
  bar = m;

  if (peek) {
    printf("BAR5 peek: INTR=0x%08x CMD=0x%08x ARG=0x%08x RESP=0x%08x\n",
           rd(OFF_INTR), rd(OFF_CMD), rd(OFF_ARG), rd(OFF_RESP));
    munmap(m, BAR_SIZE); close(fd); return 0;
  }

  printf("send cmd=0x%x (%%u=%u) arg=%u ...\n", cmd, cmd, arg);
  ret = smu_exec(cmd, arg, &out, 2000);
  printf("  ret=%d  RESP=%u  OUT=%u\n", ret, rd(OFF_RESP), out);
  munmap(m, BAR_SIZE); close(fd);
  return ret ? 1 : 0;
}
