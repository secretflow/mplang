configure_file(${CMAKE_CURRENT_SOURCE_DIR}/main.cpp.in ${CMAKE_CURRENT_BINARY_DIR}/main.cpp @ONLY)

add_executable(mplang-opt ${CMAKE_CURRENT_BINARY_DIR}/main.cpp)
target_link_libraries(mplang-opt PRIVATE
  MLIROptLib
  MLIRParser
  MLIRAsmParser
  MLIRArithDialect
  MLIRFuncDialect
  MPLANGMPIR
  MPLANGPHE
  MPLANGTransforms
)
install(TARGETS mplang-opt RUNTIME DESTINATION bin)
