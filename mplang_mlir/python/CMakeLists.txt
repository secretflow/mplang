# Python bindings for Mpir dialect

# Declare Python sources for Mpir dialect
declare_mlir_python_sources(MpirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    mplang_mlir/__init__.py
    mplang_mlir/dialects/mpir.py
    mplang_mlir/dialects/_mpir_ops_ext.py
)

# Declare dialect bindings - this will auto-generate _mpir_ops_gen.py
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MpirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mplang_mlir"
  TD_FILE ${PROJECT_SOURCE_DIR}/include/mplang/Dialect/Mpir/MpirOps.td
  SOURCES dialects/mpir.py
  DIALECT_NAME mpir
)

# Declare Python C++ extension module
declare_mlir_python_extension(MplangPythonExtension
  MODULE_NAME _mplang
  ADD_TO_PARENT MpirPythonSources
  SOURCES
    MpirExtension.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRMplangCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

# Create consolidated Python modules
declare_mlir_python_sources(MpirPythonSources.Dialects
  ADD_TO_PARENT MpirPythonSources
)

add_mlir_python_modules(MplangPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mplang_mlir"
  INSTALL_PREFIX "python_packages/mplang_mlir"
  DECLARED_SOURCES
    MpirPythonSources
    MLIRPythonSources
    MLIRPythonExtension.Core
)
