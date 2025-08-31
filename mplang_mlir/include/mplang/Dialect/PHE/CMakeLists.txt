set(LLVM_TARGET_DEFINITIONS PHE.td)
mlir_tablegen(PHEDialect.h.inc -gen-dialect-decls -dialect=phe)
mlir_tablegen(PHEDialect.cpp.inc -gen-dialect-defs -dialect=phe)
add_public_tablegen_target(MPLANGPHETableGen)

set(LLVM_TARGET_DEFINITIONS PHETypes.td)
mlir_tablegen(PHETypes.h.inc -gen-typedef-decls)
mlir_tablegen(PHETypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MPLANGPHETypesGen)

set(LLVM_TARGET_DEFINITIONS PHEOps.td)
mlir_tablegen(PHEOps.h.inc -gen-op-decls)
mlir_tablegen(PHEOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MPLANGPHEOpsGen)
