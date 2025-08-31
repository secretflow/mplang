configure_file(Peval.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/Peval.cpp @ONLY)

add_mlir_library(MPLANGTransforms
  ${CMAKE_CURRENT_BINARY_DIR}/Peval.cpp
  Passes.cpp
  PARTIAL_SOURCES_INTENDED

  DEPENDS
    MPLANGPassIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRRewrite
    MLIRTransforms
    MPLANGMPIR
)

add_dependencies(MPLANGTransforms MPLANGPassIncGen)

target_compile_definitions(MPLANGTransforms PRIVATE MPLANG_HAS_MLIR=1)
