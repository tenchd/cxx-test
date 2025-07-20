use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use cxx::Vector;





#[cxx::bridge]
mod ffi {
    #[derive(Debug)]
    struct Shared {
        v: f64,
    }

    struct SharedInt {
        v: u64,
    }

    // keeps track of outer dimension of flattened vector to rebuild safely in c++
    struct FlattenedVec {
        vec: Vec<Shared>,
        outer_length: SharedInt,
        rows: SharedInt,
    }

    unsafe extern "C++" {
        include!("cxx-test/include/example.h");

        fn f(elements: Vec<Shared>) -> Vec<Shared>;

        fn go(shared_jl_cols: FlattenedVec);
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
        println!("{}", col.len());
        jl_cols.push(col);
    }
    jl_cols
}

fn read_vecs_from_file_flat(filename: String) -> ffi::FlattenedVec {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut jl_vec: Vec<ffi::Shared> = vec![];
    let mut line_length: u64 = 0; 
    let mut first: bool = true;
    let mut line_counter = 0;

    for line in reader.lines() {
        line_counter += 1;
        let mut col: Vec<ffi::Shared> = line.expect("uh oh").split(",")
                                        .map(|x| x.trim().parse::<f64>().unwrap())
                                        .map(|v| ffi::Shared { v })
                                        .collect();
        if first {
            line_length = col.len().try_into().unwrap();
        }
        let current_line_length= col.len().try_into().unwrap();
        assert_eq!(line_length, current_line_length);
        jl_vec.append(&mut col);
    }
    //println!("line length = {}, num_lines = {}", line_length, line_counter);

    let mut jl_cols_flat = ffi::FlattenedVec{vec: jl_vec, outer_length: ffi::SharedInt{v: line_length}, rows: ffi::SharedInt {v: line_counter}};
    jl_cols_flat
}


fn main() {
    // let shared = |v| ffi::Shared { v };
    // let elements = vec![shared(3.0), shared(2.0), shared(1.0)];
    // let output = ffi::f(elements);
    // for i in output {
    //     println!("{}", i.v);
    // }

    let filename = "data/fake_jl_multi.csv".to_string();

    let shared_jl_cols_flat = read_vecs_from_file_flat(filename);

    // let shared = |v| ffi::Shared { v };
    // let new_elements = vec![shared(3.0), shared(2.0), shared(1.0), shared(4.0)];

    //let testvec = ffi::FlattenedVec {vec: new_elements, outer_length: ffi::SharedInt {v: 1},};
    ffi::go(shared_jl_cols_flat);
}