add_executable(mpir-opt mpir-opt.cpp)
target_link_libraries(mpir-opt PRIVATE
  MLIRMPIR
  MLIRIR
  MLIRParser
  MLIROptLib
  MLIRSupport
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTensorDialect
)

# Ensure MLIR headers are available to this tool
target_include_directories(mpir-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/lib/Dialect/Mpir
)

install(TARGETS mpir-opt RUNTIME DESTINATION bin)
