
#[cxx::bridge]
mod ffi {
    struct Shared {
        v: u32,
    }

    unsafe extern "C++" {
        include!("cxx-test/include/example.h");
//        include!("cxx-test/include/pre_process.hpp");
//        include!("cxx-test/include/auxilliary.hpp");
//        include!("cxx-test/include/custom_cg.hpp");

        fn f(elements: Vec<Shared>) -> Vec<Shared>;
    }
}

fn main() {
    let shared = |v| ffi::Shared { v };
    let elements = vec![shared(3), shared(2), shared(1)];
    let output = ffi::f(elements);
    for i in output {
        println!("{}", i.v);
    }
}