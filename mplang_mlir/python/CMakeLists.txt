# Python bindings for Mplang dialect

# Declare Python sources for Mplang dialect
declare_mlir_python_sources(MplangPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT MLIRPythonSources
  SOURCES
    mplang_mlir/__init__.py
    mplang_mlir/dialects/mplang.py
    mplang_mlir/dialects/_mplang_ops_ext.py
)

# Declare dialect bindings - this will auto-generate _mplang_ops_gen.py
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT MplangPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mplang_mlir"
  TD_FILE ${PROJECT_SOURCE_DIR}/include/mplang/Dialect/Mplang/MplangOps.td
  SOURCES dialects/mplang.py
  DIALECT_NAME mplang
)

# Declare Python C++ extension module
declare_mlir_python_extension(MplangPythonExtension
  MODULE_NAME _mplang
  ADD_TO_PARENT MplangPythonSources
  SOURCES
    MplangExtension.cpp
  EMBED_CAPI_LINK_LIBS
    MLIRMplangCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

# Create consolidated Python modules
declare_mlir_python_sources(MplangPythonSources.Dialects
  ADD_TO_PARENT MplangPythonSources
)

add_mlir_python_modules(MplangPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/mplang_mlir"
  INSTALL_PREFIX "python_packages/mplang_mlir"
  DECLARED_SOURCES
    MplangPythonSources
    MLIRPythonSources
    MLIRPythonExtension.Core
)
