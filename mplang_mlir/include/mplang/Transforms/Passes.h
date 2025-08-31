// Copyright 2025 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>

// Generated pass declarations live in the library TU; clients shouldn't include
// them here.

namespace mlir {
class Pass;
} // namespace mlir

namespace mplang {

// Creates the MPIR peval pass.
std::unique_ptr<mlir::Pass> createMPIRPevalPass();

// Registers all MPLANG passes with the global registry.
void registerMPLANGPasses();

} // namespace mplang
