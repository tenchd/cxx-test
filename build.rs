fn main() {
    cxx_build::bridge("src/main.rs")  // returns a cc::Build
        .file("src/example.cc")
        .std("c++20")
        .compile("cxxbridge-test");

    println!("cargo:rerun-if-changed=src/main.rs");
    println!("cargo:rerun-if-changed=src/example.cc");
    println!("cargo:rerun-if-changed=include/example.h");
}