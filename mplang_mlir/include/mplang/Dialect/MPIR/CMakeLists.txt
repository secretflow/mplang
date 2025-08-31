set(LLVM_TARGET_DEFINITIONS MPIR.td)
mlir_tablegen(MPIRDialect.h.inc -gen-dialect-decls -dialect=mpir)
mlir_tablegen(MPIRDialect.cpp.inc -gen-dialect-defs -dialect=mpir)
add_public_tablegen_target(MPLANGMPIRTableGen)

set(LLVM_TARGET_DEFINITIONS MPIRTypes.td)
mlir_tablegen(MPIRTypes.h.inc -gen-typedef-decls)
mlir_tablegen(MPIRTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MPLANGMPIRTypesGen)

set(LLVM_TARGET_DEFINITIONS MPIROps.td)
mlir_tablegen(MPIROps.h.inc -gen-op-decls)
mlir_tablegen(MPIROps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MPLANGMPIROpsGen)
