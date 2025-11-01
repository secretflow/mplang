add_executable(mplang-opt mplang-opt.cpp)
target_link_libraries(mplang-opt PRIVATE
  MLIRMPLANG
  MLIRIR
  MLIRParser
  MLIROptLib
  MLIRSupport
)

# Ensure MLIR headers are available to this tool
target_include_directories(mplang-opt PRIVATE
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
  ${CMAKE_BINARY_DIR}/lib/Dialect/MPLANG
)

install(TARGETS mplang-opt RUNTIME DESTINATION bin)
