add_mlir_public_c_api_library(MLIRMplangCAPI
  Dialects.cpp
  Registration.cpp

  LINK_LIBS PUBLIC
    MLIRMplang
    MLIRCAPIIR
)
