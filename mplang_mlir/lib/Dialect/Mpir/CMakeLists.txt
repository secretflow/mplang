# Mpir dialect library

set(MLIR_MPIR_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_MPIR_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
set(MLIR_MPIR_TD_DIR ${MLIR_MPIR_INC_DIR}/mplang/Dialect/Mpir)

# TableGen: generate ops/types/dialect decls & defs
set(LLVM_TARGET_DEFINITIONS ${MLIR_MPIR_TD_DIR}/MpirOps.td)
mlir_tablegen(MpirOps.h.inc -gen-op-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})
mlir_tablegen(MpirOps.cpp.inc -gen-op-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPIR_TD_DIR}/MpirTypes.td)
mlir_tablegen(MpirTypes.h.inc -gen-typedef-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})
mlir_tablegen(MpirTypes.cpp.inc -gen-typedef-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPIR_TD_DIR}/MpirDialect.td)
mlir_tablegen(MpirDialect.h.inc -gen-dialect-decls -dialect=mpir -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})
mlir_tablegen(MpirDialect.cpp.inc -gen-dialect-defs -dialect=mpir -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPIR_TD_DIR})
add_public_tablegen_target(MLIRMPIRIncGen)

add_mlir_library(MLIRMPIR
  MpirDialect.cpp
  MpirOps.cpp
  MpirTypes.cpp

  DEPENDS
    MLIRMPIRIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)

# Public includes: generated headers (binary dir) and repo include/
target_include_directories(MLIRMPIR
  PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MLIR_MPIR_INC_DIR}
)
