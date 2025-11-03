add_mlir_public_c_api_library(MLIRMpirCAPI
  Dialects.cpp
  Registration.cpp

  LINK_LIBS PUBLIC
    MLIRMpir
    MLIRCAPIIR
)
