// RUN: mplang-opt %s | FileCheck %s

// CHECK-LABEL: func @phe_basic
func.func @phe_basic(%x: i32) {
  // CHECK: phe.encrypt
  %c = "phe.encrypt"(%x) : (i32) -> !phe.cipher<i32, []>
  // CHECK: phe.add
  %s = "phe.add"(%c, %c) : (!phe.cipher<i32, []>, !phe.cipher<i32, []>) -> !phe.cipher<i32, []>
  // CHECK: phe.mul
  %m = "phe.mul"(%s, %c) : (!phe.cipher<i32, []>, !phe.cipher<i32, []>) -> !phe.cipher<i32, []>
  // CHECK: phe.decrypt
  %p = "phe.decrypt"(%m) {out_type = i32} : (!phe.cipher<i32, []>) -> i32
  return
}
