use std::fs::File;
use std::io::{self, prelude::*, BufReader};

use cxx::Vector;

use sprs::{CsMat};

//use crate::ffi::no_sharing;





#[cxx::bridge]
mod ffi {

    struct FlattenedVec {
        vec: Vec<f64>,
        num_cols: usize,
        num_rows: usize,
    }

    unsafe extern "C++" {
        include!("cxx-test/include/example.h");

        //fn f(elements: Vec<Shared>) -> Vec<Shared>;

        fn go(shared_jl_cols: FlattenedVec) -> FlattenedVec;

        fn sprs_test(col_ptrs: Vec<usize>, row_indices: Vec<usize>, values: Vec<f64>);
    }
}

// fn read_vecs_from_file(filename: String) -> Vec<Vec<ffi::Shared>>{
//     let file = File::open(filename).unwrap();
//     let reader = BufReader::new(file);

//     let mut jl_cols: Vec<Vec<ffi::Shared>> = vec![];

//     let column: usize = 0;
//     for line in reader.lines() {
//         let col: Vec<ffi::Shared> = line.expect("uh oh").split(",")
//                                         .map(|x| x.trim().parse::<f64>().unwrap())
//                                         .map(|v| ffi::Shared { v })
//                                         .collect();
//         println!("{}", col.len());
//         jl_cols.push(col);
//     }
//     jl_cols
// }

fn read_vecs_from_file_flat(filename: String) -> ffi::FlattenedVec {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut jl_vec: Vec<f64> = vec![];
    let mut line_length: usize = 0; 
    let mut first: bool = true;
    let mut line_counter: usize = 0;

    for line in reader.lines() {
        line_counter += 1;
        let mut col: Vec<f64> = line.expect("uh oh").split(",")
                                        .map(|x| x.trim().parse::<f64>().unwrap())
                                        //.map(|v| ffi::Shared { v })
                                        .collect();
        if first {
            line_length = col.len().try_into().unwrap();
        }
        let current_line_length= col.len().try_into().unwrap();
        assert_eq!(line_length, current_line_length);
        jl_vec.append(&mut col);
    }
    //println!("line length = {}, num_lines = {}", line_length, line_counter);

    let mut jl_cols_flat = ffi::FlattenedVec{vec: jl_vec, num_cols: line_counter, num_rows: line_length};
    jl_cols_flat
}


fn main() {

    let filename = "data/fake_jl_multi.csv".to_string();

    let shared_jl_cols_flat = read_vecs_from_file_flat(filename);
    let m = shared_jl_cols_flat.num_cols;
    let n = shared_jl_cols_flat.num_rows;

    // let shared = |v| ffi::Shared { v };
    // let new_elements = vec![shared(3.0), shared(2.0), shared(1.0), shared(4.0)];

    //let testvec = ffi::FlattenedVec {vec: new_elements, outer_length: ffi::SharedInt {v: 1},};

    //let result = ffi::go(shared_jl_cols_flat);
    
    //println!("{}",result.vec[0]);

    let a = CsMat::new_csc((3, 3),
                       vec![0, 2, 4, 5],
                       vec![0, 1, 0, 2, 2],
                       vec![1., 2., 3., 4., 5.]);

    let col_ptrs = a.indptr().as_slice().unwrap().to_vec();
    let row_indices = a.indices().to_vec();
    let values= a.data().to_vec();
    println!("col_ptrs in rust: {:?}", col_ptrs);
    println!("row_indices in rust: {:?}", row_indices);
    println!("values in rust: {:?}", values);
    ffi::sprs_test(col_ptrs, row_indices, values);
}