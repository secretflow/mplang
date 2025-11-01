//===- MPLANGOps.td ---------------------------------------------*- tablegen -*-===//
// Minimal MPLANG ops ODS declarations: eval (region) and yield.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "MPLANGDialect.td"

def MPLANG_YieldOp : Op<MPLANG_Dialect, "yield", [Terminator]> {
  let summary = "Yield results from peval region";
  let arguments = (ins Variadic<AnyType>:$results);
  // Explicitly print operand types to make the format buildable for tblgen
  let assemblyFormat = "operands attr-dict `:` type($results)";

  // Provide a default no-arg builder so it can be used as an implicit
  // terminator for regions that are left empty.
  let builders = [
    OpBuilder<(ins), [{ /* default empty builder */ }]>
  ];
}

def MPLANG_EvalOp : Op<MPLANG_Dialect, "eval",
    [SingleBlockImplicitTerminator<"::mplang::YieldOp">]> {
  let summary = "SPMD per-party execution (region form)";
  let description = [{
    Executes a per-party region on a specified runtime mask of parties.
    Inputs/outputs are distributed values (to be refined by type parameters).
  }];

  let arguments = (ins Variadic<AnyType>:$inputs);
  let results   = (outs Variadic<AnyType>:$outs);
  let regions   = (region AnyRegion:$body);

  let assemblyFormat = "`(` $inputs `)` attr-dict `(` $body `)` `:` functional-type($inputs, $outs)";
}
