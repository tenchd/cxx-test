use std::path::Path;

fn main() {
    let fast_mtx_path = Path::new("/global/u1/d/dtench/cholesky/fast_matrix_market/include");

    cxx_build::bridge("src/main.rs")  // returns a cc::Build
        .file("src/example.cc")
        .include(fast_mtx_path)
        .std("c++20")
//        .flags("")
        .compile("cxxbridge-test");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/driver_local.cpp");
    println!("cargo:rerun-if-changed=src/example.cc");
    println!("cargo:rerun-if-changed=include/example.h");
    println!("cargo:rerun-if-changed=include/auxilliary.hpp");
    println!("cargo:rerun-if-changed=include/custom_cg.hpp");
    println!("cargo:rerun-if-changed=include/pre_process.hpp");
}