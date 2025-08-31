# Resolve FileCheck from the LLVM/MLIR install tree (built by setup script).
# Always try to refresh from MLIR_DIR to avoid stale cache values.
# MLIR_DIR typically looks like <root>/lib/cmake/mlir
get_filename_component(_mlir_cmake_dir "${MLIR_DIR}" DIRECTORY)
get_filename_component(_llvm_lib_dir "${_mlir_cmake_dir}" DIRECTORY)
get_filename_component(_llvm_root "${_llvm_lib_dir}" DIRECTORY)
set(_llvm_tools_bin "${_llvm_root}/bin")

# Clear any cached NOTFOUND and search preferred location first.
unset(FILECHECK_EXE CACHE)
find_program(FILECHECK_EXE NAMES FileCheck HINTS "${_llvm_tools_bin}")

# Fallback to PATH search if still not found.
if(NOT FILECHECK_EXE)
  find_program(FILECHECK_EXE NAMES FileCheck)
endif()

function(add_mplang_filecheck_test name file pattern)
  if(NOT FILECHECK_EXE)
    message(WARNING "FileCheck not found; skipping test ${name}")
    return()
  endif()
  add_test(NAME mplang-${name}
    COMMAND bash -c "${CMAKE_BINARY_DIR}/tools/mplang-opt/mplang-opt ${pattern} ${file} | \"${FILECHECK_EXE}\" ${file}"
  )
  # Ensure a sane PATH for any subprocesses, though FileCheck is absolute.
  set_tests_properties(mplang-${name} PROPERTIES ENVIRONMENT "PATH=${CMAKE_BINARY_DIR}/tools/mplang-opt:${CMAKE_BINARY_DIR}/bin:$ENV{PATH}")
endfunction()

add_mplang_filecheck_test(mpir-peval "${CMAKE_SOURCE_DIR}/test/mlir/mpir/peval.mlir" "--mpir-peval")

# Plain FileCheck tests without a specific pass pipeline.
  add_mplang_filecheck_test(mpir-ops "${CMAKE_SOURCE_DIR}/test/mlir/mpir/ops.mlir" "")
  ## Temporarily disable until textual assembly is implemented for these dialect pieces.
  # add_mplang_filecheck_test(mpir-cf  "${CMAKE_SOURCE_DIR}/test/mlir/mpir/cf.mlir"  "")
  # add_mplang_filecheck_test(phe-ops  "${CMAKE_SOURCE_DIR}/test/mlir/phe/ops.mlir"  "")
