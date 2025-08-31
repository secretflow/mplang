configure_file(MPIRDialect.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/MPIRDialectImpl.cpp @ONLY)

add_mlir_dialect_library(MPLANGMPIR
  ${CMAKE_CURRENT_BINARY_DIR}/MPIRDialectImpl.cpp

  DEPENDS
    MPLANGMPIRTableGen
    MPLANGMPIRTypesGen
    MPLANGMPIROpsGen

  LINK_LIBS PUBLIC
    MLIRIR
)

target_compile_definitions(MPLANGMPIR PRIVATE MPLANG_HAS_MLIR=1)
