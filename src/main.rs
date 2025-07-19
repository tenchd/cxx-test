use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use cxx::Vector;





#[cxx::bridge]
mod ffi {
    #[derive(Debug)]
    struct Shared {
        v: f64,
    }

    unsafe extern "C++" {
        include!("cxx-test/include/example.h");
//        include!("cxx-test/include/pre_process.hpp");
//        include!("cxx-test/include/auxilliary.hpp");
//        include!("cxx-test/include/custom_cg.hpp");

        fn f(elements: Vec<Shared>) -> Vec<Shared>;

        fn go(shared_jl_cols: Vec<Vec<Shared>>);
    }
}

fn read_vecs_from_file(filename: String) -> Vec<Vec<ffi::Shared>>{
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut jl_cols: Vec<Vec<ffi::Shared>> = vec![];

    let column: usize = 0;
    for line in reader.lines() {
        let col: Vec<ffi::Shared> = line.expect("uh oh").split(",")
                                        .map(|x| x.trim().parse::<f64>().unwrap())
                                        .map(|v| ffi::Shared { v })
                                        .collect();
        jl_cols.push(col);
    }
    jl_cols
}


fn main() {
    let shared = |v| ffi::Shared { v };
    let elements = vec![shared(3.0), shared(2.0), shared(1.0)];
    let output = ffi::f(elements);
    for i in output {
        println!("{}", i.v);
    }

    let filename = "data/fake_jl_multi_small.csv".to_string();

    let shared_jl_cols = read_vecs_from_file(filename);
}