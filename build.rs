use std::path::Path;

fn main() {
    let fast_mtx_path = Path::new("/global/u1/d/dtench/cholesky/fast_matrix_market/include");

    cxx_build::bridge("src/main.rs")  // returns a cc::Build
        .compiler("g++")
        .file("src/example.cc")
        .include(fast_mtx_path)
        .std("c++20")
//        .flags(["-O3", "-fopenmp", "-L", "-lm", "-lmkl_intel_lp64", "-lmkl_intel_thread", "-lmkl_core", "-liomp5", "-lpthread"])
        .flag("-O3")
        .flag("-fopenmp")
        .flag("-w")
//        .flag("-g")
        .compile("cxxbridge-test");

    println!("cargo::rustc-link-lib=m");
    println!("cargo::rustc-link-lib=mkl_intel_lp64");
    println!("cargo::rustc-link-lib=mkl_intel_thread");
    println!("cargo::rustc-link-lib=mkl_core");
    println!("cargo::rustc-link-lib=iomp5");
    println!("cargo::rustc-link-lib=pthread");
    //println!("cargo::rustc-link-arg=-fopenmp");


    println!("cargo:rerun-if-changed=src/main.rs");
//    println!("cargo:rerun-if-changed=src/driver_local.cpp");
    println!("cargo:rerun-if-changed=src/example.cc");
    println!("cargo:rerun-if-changed=include/example.h");
    println!("cargo:rerun-if-changed=include/auxilliary.hpp");
    println!("cargo:rerun-if-changed=include/custom_cg.hpp");
    println!("cargo:rerun-if-changed=include/pre_process.hpp");
}