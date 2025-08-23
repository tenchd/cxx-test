#![allow(unused)]
#![feature(test)]
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use cxx::Vector;
use sprs::{CsMat,CsVecI};
extern crate fasthash;
extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
extern crate rand;
extern crate test;

mod utils;
mod jl_sketch;
mod sparsifier;
mod stream;
mod tests;

use utils::{read_mtx, write_mtx, write_csv, read_vecs_from_file_flat, make_fake_jl_col,create_trivial_rhs};
use jl_sketch::{jl_sketch_sparse,jl_sketch_sparse_blocked};
use sparsifier::{Sparsifier,Triplet};
use stream::InputStream;
use crate::ffi::FlattenedVec;


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

        fn run_solve_lap(shared_jl_cols: FlattenedVec, rust_col_ptrs: Vec<i32>, rust_row_indices: Vec<i32>, rust_values: Vec<f64>, num_nodes:i32) -> FlattenedVec;
    }
}



fn jl_sketch_dataset(input_filename: &str, output_filename: &str, jl_factor: f64, seed: u64){
    //be careful about jl sketch dimensions now that i'm adding a row and col to the laplacian. check this later
    let input_csc = read_mtx(input_filename, false);

    let output_csc = jl_sketch_sparse(&input_csc, jl_factor, seed);
    let dense_output = output_csc.to_dense().reversed_axes();
    write_csv(output_filename, &dense_output);

}

// currently assumes that you don't need to manage diagonals of input matrix. fix this later
fn precondition_and_solve(input_filename: &str, sketch_filename: &str, seed: u64, jl_factor: f64, block_rows: usize, block_cols: usize, display: bool, add_node: bool) -> FlattenedVec {
    //let filename = "data/fake_jl_multi.csv".to_string();

    let input_csc = read_mtx(input_filename, add_node);
    println!("{}", input_csc.outer_dims());

    //make sure diagonals are nonzero; generalize this later
    // for i in input_csc.diag_iter(){
    //     assert!(*i.unwrap() != 0.0 as f64)
    // }

    let n: usize = input_csc.cols();
    let m: usize = input_csc.rows();
    assert_eq!(n,m);

    //let seed: u64 = 1;
    //let jl_factor: f64 = 1.5;
    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize;
    println!("output matrix should be {}x{}", n, jl_dim);

    // not used at the moment; need to convert it to flattenedvec so i can pass it to c++. try the to_dense function maybe?
    //let mut sketch_sparse_format: CsMat<f64> = CsMat::zero((jl_dim,n)).transpose_into();
    //println!("{}", sketch_sparse_format.outer_dims());

    //jl_sketch_sparse_blocked(&input_csc, &mut sketch_sparse_format, jl_dim, seed, block_rows, block_cols, display);
    //write_mtx("real_jl_sketch", &sketch_sparse_format);
    
    let shared_jl_cols_flat = read_vecs_from_file_flat(sketch_filename);
    let m: i32 = shared_jl_cols_flat.num_cols.try_into().unwrap();
    let n: i32 = shared_jl_cols_flat.num_rows.try_into().unwrap();
    println!("output matrix actually is {}x{}", n, m);    

    // let input_col_ptrs = input_csc.indptr().as_slice().unwrap().to_vec();
    // let input_row_indices = input_csc.indices().to_vec();

    let temp_input_col_ptrs = input_csc.indptr().as_slice().unwrap().to_vec();
    let input_col_ptrs: Vec<i32> = temp_input_col_ptrs.into_iter().map(|x| x as i32).collect();
    let temp_input_row_indices = input_csc.indices().to_vec();
    let input_row_indices: Vec<i32> = temp_input_row_indices.into_iter().map(|x| x as i32).collect();

    let input_values = input_csc.data().to_vec();

    println!("input col_ptrs size in rust: {:?}. first value: {}", input_col_ptrs.len(), input_col_ptrs[0]);
    println!("input row_indices size in rust: {:?}. first value: {}", input_row_indices.len(), input_row_indices[0]);
    println!("input values size in rust: {:?}. first value: {}", input_values.len(), input_values[0]);
    println!("nodes in input csc: {}, {}", input_csc.cols(), input_csc.rows());
    //ffi::sprs_correctness_test(input_col_ptrs, input_row_indices, input_values);
    ffi::run_solve_lap(shared_jl_cols_flat, input_col_ptrs, input_row_indices, input_values, n)
}



fn solve_test() {

    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;
    let block_rows: usize = 100;
    let block_cols: usize = 15000;
    let display: bool = false;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = true;

    //let stream = InputStream::new("data/cage3.mtx");
    //stream.run_stream(epsilon, beta_constant, row_constant, verbose);


    let sketch_filename = "data/fake_jl_multi.csv";
    let input_filename = "/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx";
    let add_node = false;

    //let sketch_filename = "data/virus_jl_sketch.csv";
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    //let add_node = true;


    let solution = precondition_and_solve(input_filename, sketch_filename, seed, jl_factor, block_rows, block_cols, display, add_node);

    println!("solution has {} cols, {} rows, and initial value {}", solution.num_cols, solution.num_rows, solution.vec[0]);

}

fn lap_test(input_filename: &str) {
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;
    let block_rows: usize = 100;
    let block_cols: usize = 15000;
    let display: bool = false;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = false;

    let add_node = false;

    let stream = InputStream::new(input_filename, add_node);
    stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed);
}

fn main() {
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    let input_filename = "/global/u1/d/dtench/rust_spars/cxx-test/data/cage3.mtx";
    lap_test(input_filename);
    //solve_test();

    //println!("{:?}", create_trivial_rhs(10));
}