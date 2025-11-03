// RUN: mpir-opt %s | mpir-opt | FileCheck %s

module {
  // CHECK-LABEL: @test_table_type
  func.func @test_table_type(%arg0: !mpir.table<["id", "age"], [i64, i32]>)
                             -> !mpir.table<["id", "age"], [i64, i32]> {
    // CHECK: %arg0 : !mpir.table<[id, age], [i64, i32]>
    return %arg0 : !mpir.table<["id", "age"], [i64, i32]>
  }

  // CHECK-LABEL: @test_mp_table
  func.func @test_mp_table(%arg0: !mpir.mp<!mpir.table<["name", "score"], [i64, f64]>, 3>)
                           -> !mpir.mp<!mpir.table<["name", "score"], [i64, f64]>, 3> {
    // CHECK: %arg0 : !mpir.mp<!mpir.table<[name, score], [i64, f64]>, 3>
    return %arg0 : !mpir.mp<!mpir.table<["name", "score"], [i64, f64]>, 3>
  }

  // CHECK-LABEL: @test_encoded_table_parquet
  func.func @test_encoded_table_parquet(%arg0: !mpir.enc<!mpir.table<["id"], [i64]>, "parquet">)
                                        -> !mpir.enc<!mpir.table<["id"], [i64]>, "parquet"> {
    // CHECK: %arg0 : !mpir.enc<!mpir.table<[id], [i64]>, parquet>
    return %arg0 : !mpir.enc<!mpir.table<["id"], [i64]>, "parquet">
  }

  // CHECK-LABEL: @test_encoded_table_csv
  func.func @test_encoded_table_csv(%arg0: !mpir.enc<!mpir.table<["col1", "col2"], [f32, f32]>, "csv", {delimiter = ",", header = true}>)
                                    -> !mpir.enc<!mpir.table<["col1", "col2"], [f32, f32]>, "csv", {delimiter = ",", header = true}> {
    // CHECK: %arg0 : !mpir.enc<!mpir.table<[col1, col2], [f32, f32]>, csv, {delimiter = ",", header = true}>
    return %arg0 : !mpir.enc<!mpir.table<["col1", "col2"], [f32, f32]>, "csv", {delimiter = ",", header = true}>
  }

  // CHECK-LABEL: @test_mp_encoded_table
  func.func @test_mp_encoded_table(%arg0: !mpir.mp<!mpir.enc<!mpir.table<["id", "value"], [i64, f32]>, "parquet">, 1>)
                                   -> !mpir.mp<!mpir.enc<!mpir.table<["id", "value"], [i64, f32]>, "parquet">, 1> {
    // CHECK: %arg0 : !mpir.mp<!mpir.enc<!mpir.table<[id, value], [i64, f32]>, parquet>, 1>
    return %arg0 : !mpir.mp<!mpir.enc<!mpir.table<["id", "value"], [i64, f32]>, "parquet">, 1>
  }

  // CHECK-LABEL: @test_encrypted_table_storage
  func.func @test_encrypted_table_storage(%arg0: !mpir.mp<!mpir.enc<!mpir.table<["ssn", "balance"], [i64, f64]>, "aes-gcm", {key_size = 256}>, 1>)
                                          -> !mpir.mp<!mpir.enc<!mpir.table<["ssn", "balance"], [i64, f64]>, "aes-gcm", {key_size = 256}>, 1> {
    // CHECK: %arg0 : !mpir.mp<!mpir.enc<!mpir.table<[ssn, balance], [i64, f64]>, aes-gcm, {key_size = 256 : i64}>, 1>
    return %arg0 : !mpir.mp<!mpir.enc<!mpir.table<["ssn", "balance"], [i64, f64]>, "aes-gcm", {key_size = 256}>, 1>
  }

  // CHECK-LABEL: @test_single_column_table
  func.func @test_single_column_table(%arg0: !mpir.table<["data"], [tensor<10xf32>]>)
                                      -> !mpir.table<["data"], [tensor<10xf32>]> {
    // CHECK: %arg0 : !mpir.table<[data], [tensor<10xf32>]>
    return %arg0 : !mpir.table<["data"], [tensor<10xf32>]>
  }

  // CHECK-LABEL: @test_many_columns_table
  func.func @test_many_columns_table(%arg0: !mpir.table<["c1", "c2", "c3", "c4", "c5"], [i32, i64, f32, f64, i1]>)
                                     -> !mpir.table<["c1", "c2", "c3", "c4", "c5"], [i32, i64, f32, f64, i1]> {
    // CHECK: %arg0 : !mpir.table<[c1, c2, c3, c4, c5], [i32, i64, f32, f64, i1]>
    return %arg0 : !mpir.table<["c1", "c2", "c3", "c4", "c5"], [i32, i64, f32, f64, i1]>
  }
}
