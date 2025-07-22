use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use cxx::Vector;
use sprs::{CsMat};

mod utils;
use utils::read_mtx;

use crate::ffi::FlattenedVec;

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

        fn sprs_correctness_test(col_ptrs: Vec<i32>, row_indices: Vec<i32>, values: Vec<f64>);

        fn run_solve_lap(shared_jl_cols: FlattenedVec, rust_col_ptrs: Vec<i32>, rust_row_indices: Vec<i32>, rust_values: Vec<f64>) -> FlattenedVec;
    }
}

// reads a jl_sketch vec of vecs from a file and 
fn read_vecs_from_file_flat(filename: &str) -> ffi::FlattenedVec {
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


// currently assumes that you don't need to manage diagonals of input matrix. fix this later
fn precondition_and_solve(input_filename: &str, sketch_filename: &str) -> FlattenedVec {
    //let filename = "data/fake_jl_multi.csv".to_string();

    let shared_jl_cols_flat = read_vecs_from_file_flat(sketch_filename);
    let m = shared_jl_cols_flat.num_cols;
    let n = shared_jl_cols_flat.num_rows;

    let input_csc = read_mtx(input_filename);
    
    for i in input_csc.diag_iter(){
        assert!(*i.unwrap() != 0.0 as f64)
    }

    let temp_input_col_ptrs = input_csc.indptr().as_slice().unwrap().to_vec();
    let input_col_ptrs: Vec<i32> = temp_input_col_ptrs.into_iter().map(|x| x as i32).collect();
    let temp_input_row_indices = input_csc.indices().to_vec();
    let input_row_indices: Vec<i32> = temp_input_row_indices.into_iter().map(|x| x as i32).collect();
    let input_values = input_csc.data().to_vec();
    //let input_values = temp_input_values.into_iter().map(|x| x as u64).collect();
    println!("input col_ptrs size in rust: {:?}. first value: {}", input_col_ptrs.len(), input_col_ptrs[0]);
    println!("input row_indices size in rust: {:?}. first value: {}", input_row_indices.len(), input_row_indices[0]);
    println!("input values size in rust: {:?}. first value: {}", input_values.len(), input_values[0]);
    println!("nodes in input csc: {}, {}", input_csc.cols(), input_csc.rows());
    //ffi::sprs_correctness_test(input_col_ptrs, input_row_indices, input_values);
    ffi::run_solve_lap(shared_jl_cols_flat, input_col_ptrs, input_row_indices, input_values)
}

fn main() {

    let sketch_filename = "data/fake_jl_multi.csv";
    let input_filename = "/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx";

    let solution = precondition_and_solve(input_filename, sketch_filename);

    println!("solution has {} cols, {} rows, and initial value {}", solution.num_cols, solution.num_rows, solution.vec[0]);


    // let shared_jl_cols_flat = read_vecs_from_file_flat(filename);
    // let m = shared_jl_cols_flat.num_cols;
    // let n = shared_jl_cols_flat.num_rows;

    // let physics_csc = read_mtx("/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx");
    
    // for i in physics_csc.diag_iter(){
    //     assert!(*i.unwrap() != 0.0 as f64)
    // }

    // let temp_phys_col_ptrs = physics_csc.indptr().as_slice().unwrap().to_vec();
    // let phys_col_ptrs: Vec<i32> = temp_phys_col_ptrs.into_iter().map(|x| x as i32).collect();
    // let temp_phys_row_indices = physics_csc.indices().to_vec();
    // let phys_row_indices: Vec<i32> = temp_phys_row_indices.into_iter().map(|x| x as i32).collect();
    // let phys_values = physics_csc.data().to_vec();
    // //let phys_values = temp_phys_values.into_iter().map(|x| x as u64).collect();
    // println!("phys col_ptrs size in rust: {:?}. first value: {}", phys_col_ptrs.len(), phys_col_ptrs[0]);
    // println!("phys row_indices size in rust: {:?}. first value: {}", phys_row_indices.len(), phys_row_indices[0]);
    // println!("phys values size in rust: {:?}. first value: {}", phys_values.len(), phys_values[0]);
    // println!("nodes in phys csc: {}, {}", physics_csc.cols(), physics_csc.rows());
    // //ffi::sprs_correctness_test(phys_col_ptrs, phys_row_indices, phys_values);
    // ffi::run_solve_lap(shared_jl_cols_flat, phys_col_ptrs, phys_row_indices, phys_values);

}