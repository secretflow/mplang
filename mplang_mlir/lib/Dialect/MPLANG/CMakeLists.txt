# MPLANG dialect library

set(MLIR_MPLANG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_MPLANG_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
set(MLIR_MPLANG_TD_DIR ${MLIR_MPLANG_INC_DIR}/mplang/Dialect/MPLANG)

# TableGen: generate ops/types/dialect decls & defs
set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MPLANGOps.td)
mlir_tablegen(MPLANGOps.h.inc -gen-op-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MPLANGOps.cpp.inc -gen-op-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MPLANGTypes.td)
mlir_tablegen(MPLANGTypes.h.inc -gen-typedef-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MPLANGTypes.cpp.inc -gen-typedef-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MPLANGDialect.td)
mlir_tablegen(MPLANGDialect.h.inc -gen-dialect-decls -dialect=mplang -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MPLANGDialect.cpp.inc -gen-dialect-defs -dialect=mplang -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
add_public_tablegen_target(MLIRMPLANGIncGen)

add_mlir_library(MLIRMPLANG
  MPLANGDialect.cpp
  MPLANGOps.cpp
  MPLANGTypes.cpp

  DEPENDS
    MLIRMPLANGIncGen

  LINK_LIBS PUBLIC
    MLIRIR
)

# Public includes: generated headers (binary dir) and repo include/
target_include_directories(MLIRMPLANG
  PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MLIR_MPLANG_INC_DIR}
)
