add_custom_target(check-mplang
  COMMAND ${CMAKE_CTEST_COMMAND} -R mplang -V
  DEPENDS mplang-opt
)

add_subdirectory(mlir)
