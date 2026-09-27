#!/usr/bin/env python3
"""Unprivileged SoC/IPU telemetry from amdgpu's gpu_metrics blob (v3.0).

Decodes struct gpu_metrics_v3_0 (kgd_pp_interface.h). Despite the name it is the
whole-APU metrics table, not just graphics: it carries the SMU's IPU fields
(IPUCLK, MPIPU, IPU power, per-column busy, IPU reads/writes), the power
breakdown, the enforced STAPM/GFX limits and the throttle-residency counters.
"""
import ctypes, sys, time

P = "/sys/class/drm/card1/device/gpu_metrics"

class Hdr(ctypes.LittleEndianStructure):
    _fields_ = [("structure_size", ctypes.c_uint16),
                ("format_revision", ctypes.c_uint8),
                ("content_revision", ctypes.c_uint8)]

class M(ctypes.LittleEndianStructure):
    _fields_ = [
        ("hdr", Hdr),
        ("temperature_gfx", ctypes.c_uint16), ("temperature_soc", ctypes.c_uint16),
        ("temperature_core", ctypes.c_uint16*16), ("temperature_skin", ctypes.c_uint16),
        ("average_gfx_activity", ctypes.c_uint16), ("average_vcn_activity", ctypes.c_uint16),
        ("average_ipu_activity", ctypes.c_uint16*8),
        ("average_core_c0_activity", ctypes.c_uint16*16),
        ("average_dram_reads", ctypes.c_uint16), ("average_dram_writes", ctypes.c_uint16),
        ("average_ipu_reads", ctypes.c_uint16), ("average_ipu_writes", ctypes.c_uint16),
        ("system_clock_counter", ctypes.c_uint64),
        ("average_socket_power", ctypes.c_uint32), ("average_ipu_power", ctypes.c_uint16),
        ("average_apu_power", ctypes.c_uint32), ("average_gfx_power", ctypes.c_uint32),
        ("average_dgpu_power", ctypes.c_uint32), ("average_all_core_power", ctypes.c_uint32),
        ("average_core_power", ctypes.c_uint16*16),
        ("average_sys_power", ctypes.c_uint16), ("stapm_power_limit", ctypes.c_uint16),
        ("current_stapm_power_limit", ctypes.c_uint16),
        ("average_gfxclk_frequency", ctypes.c_uint16), ("average_socclk_frequency", ctypes.c_uint16),
        ("average_vpeclk_frequency", ctypes.c_uint16), ("average_ipuclk_frequency", ctypes.c_uint16),
        ("average_fclk_frequency", ctypes.c_uint16), ("average_vclk_frequency", ctypes.c_uint16),
        ("average_uclk_frequency", ctypes.c_uint16), ("average_mpipu_frequency", ctypes.c_uint16),
        ("current_coreclk", ctypes.c_uint16*16),
        ("current_core_maxfreq", ctypes.c_uint16), ("current_gfx_maxfreq", ctypes.c_uint16),
        ("throttle_residency_prochot", ctypes.c_uint32), ("throttle_residency_spl", ctypes.c_uint32),
        ("throttle_residency_fppt", ctypes.c_uint32), ("throttle_residency_sppt", ctypes.c_uint32),
        ("throttle_residency_thm_core", ctypes.c_uint32), ("throttle_residency_thm_gfx", ctypes.c_uint32),
        ("throttle_residency_thm_soc", ctypes.c_uint32),
        ("time_filter_alphavalue", ctypes.c_uint32),
    ]

assert ctypes.sizeof(M) == 264, ctypes.sizeof(M)

def read():
    with open(P, "rb") as f:
        buf = f.read(ctypes.sizeof(M))
    m = M.from_buffer_copy(buf)
    return m

def line(m):
    ipu_busy = "/".join(str(x) for x in m.average_ipu_activity)
    return (f"Tgfx={m.temperature_gfx/100:5.1f} Tsoc={m.temperature_soc/100:5.1f} "
            f"skin={m.temperature_skin/100:5.1f} | "
            f"clk gfx={m.average_gfxclk_frequency} soc={m.average_socclk_frequency} "
            f"fclk={m.average_fclk_frequency} uclk={m.average_uclk_frequency} "
            f"IPU={m.average_ipuclk_frequency} MPIPU={m.average_mpipu_frequency} | "
            f"P pkg={m.average_socket_power/1000:5.1f} apu={m.average_apu_power/1000:5.1f} "
            f"gfx={m.average_gfx_power/1000:5.1f} cores={m.average_all_core_power/1000:5.1f} "
            f"IPU={m.average_ipu_power/1000:5.2f} | "
            f"act gfx={m.average_gfx_activity} ipu={ipu_busy} | "
            f"lim stapm={m.stapm_power_limit/1000:.0f}/{m.current_stapm_power_limit/1000:.0f} "
            f"gfxmax={m.current_gfx_maxfreq} | "
            f"thr spl={m.throttle_residency_spl} fppt={m.throttle_residency_fppt} "
            f"sppt={m.throttle_residency_sppt} prochot={m.throttle_residency_prochot} "
            f"thm_gfx={m.throttle_residency_thm_gfx} alpha={m.time_filter_alphavalue}")

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    for i in range(n):
        print(f"{time.monotonic():9.2f} {line(read())}")
        sys.stdout.flush()
        if i + 1 < n: time.sleep(iv)
