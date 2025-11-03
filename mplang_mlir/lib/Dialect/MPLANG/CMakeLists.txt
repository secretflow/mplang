# Mplang dialect library

set(MLIR_MPLANG_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(MLIR_MPLANG_INC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../include)
set(MLIR_MPLANG_TD_DIR ${MLIR_MPLANG_INC_DIR}/mplang/Dialect/MPLANG)

# TableGen: generate ops/types/dialect decls & defs
set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MplangOps.td)
mlir_tablegen(MplangOps.h.inc -gen-op-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MplangOps.cpp.inc -gen-op-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MplangTypes.td)
mlir_tablegen(MplangTypes.h.inc -gen-typedef-decls -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MplangTypes.cpp.inc -gen-typedef-defs -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})

set(LLVM_TARGET_DEFINITIONS ${MLIR_MPLANG_TD_DIR}/MplangDialect.td)
mlir_tablegen(MplangDialect.h.inc -gen-dialect-decls -dialect=mplang -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
mlir_tablegen(MplangDialect.cpp.inc -gen-dialect-defs -dialect=mplang -I ${MLIR_INCLUDE_DIRS} -I ${MLIR_MPLANG_TD_DIR})
add_public_tablegen_target(MLIRMPLANGIncGen)

add_mlir_library(MLIRMPLANG
  MplangDialect.cpp
  MplangOps.cpp
  MplangTypes.cpp

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
