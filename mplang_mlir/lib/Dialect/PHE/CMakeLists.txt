configure_file(PHEDialect.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/PHEDialect.cpp @ONLY)

add_mlir_dialect_library(MPLANGPHE
  ${CMAKE_CURRENT_BINARY_DIR}/PHEDialect.cpp

  DEPENDS
    MPLANGPHETableGen
    MPLANGPHETypesGen
    MPLANGPHEOpsGen

  LINK_LIBS PUBLIC
    MLIRIR
)

target_compile_definitions(MPLANGPHE PRIVATE MPLANG_HAS_MLIR=1)
