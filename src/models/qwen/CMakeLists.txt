target_sources(gufo_core PRIVATE
  ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/prompt.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/tokenizer.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/chat_template.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/oracles.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/state.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/weights.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/forward.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/attention.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/embed.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ffn.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/norm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/quant_gemm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/residual.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/rope.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/sample.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/ssm.cpp
  ${PROJECT_SOURCE_DIR}/src/models/qwen/modules/unembed.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/generator.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/mtp_reference.cpp
   ${PROJECT_SOURCE_DIR}/src/models/qwen/dflash_weights.cpp)

if (ENGINE_ENABLE_HIP)
  find_package(OpenSSL REQUIRED COMPONENTS Crypto)
  target_link_libraries(gufo_core PRIVATE OpenSSL::Crypto)
  target_sources(gufo_core PRIVATE
    ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/encoder.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/vision/device_input.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/attention.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/batched_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_fp16.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/embed.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/residual.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/unpack.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/rope.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/sample.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/norm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/gemv_quant.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/fused.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/qkv.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_decode_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_recurrence.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/ssm_row_split.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/swiglu.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_batched.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_decode_graph.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_tile.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/attention_wmma.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblas_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/hipblaslt_gemm.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/dflash_kernels.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/arena.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/atb_npu.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/batched_decode.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/decode_step.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/model_loader.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/prefill_chunk.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/ssm_replay.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/mtp/backend.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/model.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/executor.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/batch.cpp
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/dflash/backend.cpp)

  # The NPU half of the FFN split needs XRT and the amdxdna runtime. Neither is
  # a hard dependency: without them atb_npu.cpp compiles to the stub that
  # reports the path unavailable, so the same source tree builds everywhere.
  find_path(GUFO_XRT_INCLUDE_DIR xrt/xrt_bo.h HINTS /usr/include /opt/xilinx/xrt/include)
  find_library(GUFO_XRT_COREUTIL_LIBRARY xrt_coreutil
    HINTS /usr/lib /usr/lib64 /opt/xilinx/xrt/lib)
  # hsa-runtime64 is a ROCm library, so it is not on the default search path.
  find_library(GUFO_HSA_RUNTIME_LIBRARY hsa-runtime64 HINTS /opt/rocm/lib)
  if (GUFO_XRT_INCLUDE_DIR AND GUFO_XRT_COREUTIL_LIBRARY AND GUFO_HSA_RUNTIME_LIBRARY)
    target_compile_definitions(gufo_core PRIVATE GUFO_ENABLE_NPU=1)
    target_include_directories(gufo_core PRIVATE ${GUFO_XRT_INCLUDE_DIR})
    target_link_libraries(gufo_core PRIVATE
      ${GUFO_XRT_COREUTIL_LIBRARY} ${GUFO_HSA_RUNTIME_LIBRARY})
    message(STATUS "Qwen ATB NPU split enabled: XRT ${GUFO_XRT_COREUTIL_LIBRARY}")
  else()
    message(STATUS "Qwen ATB NPU split disabled: XRT or hsa-runtime64 not found")
  endif()

  # HIP helpers must share the kernel's wave size, so compile this unit
  # separately while the other Qwen kernels retain wave32.
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_wave64.hip
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/prefill_quant_wave64.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES COMPILE_OPTIONS -mwavefrontsize64)
  # Iterative ILP improves Q4/Q5 groups of sixteen without changing arithmetic.
  # Keep it isolated: other widths regress. Line tables retain source profiling
  # while avoiding an LLVM register-pressure crash with full debug information.
  set_source_files_properties(
    ${PROJECT_SOURCE_DIR}/src/models/qwen/hip/kernels/small_batch_quant16_wave64.hip
    TARGET_DIRECTORY gufo_core
    PROPERTIES COMPILE_OPTIONS
    "-mwavefrontsize64;-gline-tables-only;-Xarch_device;-mllvm=-amdgpu-sched-strategy=iterative-ilp")
endif()
