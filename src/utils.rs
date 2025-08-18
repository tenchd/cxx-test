// this file stores boring functions for the Rust side of the sparsifier implementation. 
// boring means not related to ffi, not really part of the core sparsifier logic, etc.

use sprs::{CsMat,CsMatI,TriMatI,CsVec,CsVecI};
use sprs::io::{read_matrix_market, write_matrix_market};

use ndarray::{array, Array2};
use ndarray_csv::{Array2Reader, Array2Writer};
use std::error::Error;
use csv::{ReaderBuilder, WriterBuilder};
use std::fs::File;
use std::io::{self, prelude::*, BufReader};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use approx::AbsDiffEq;

use crate::sparsifier::{Sparsifier,Triplet};
use crate::ffi;

pub fn read_mtx(filename: &str, add_node: bool) -> CsMatI<f64, i32>{
    let trip = read_matrix_market::<f64, i32, &str>(filename).unwrap();
    // for i in trip.triplet_iter() {
    //     println!("{:?}", i);
    // }

    assert_eq!(trip.cols(), trip.rows());
    if add_node {
        let num_nodes = trip.cols();
        let mut trip_fixed = TriMatI::<f64, i32>::new((num_nodes+1, num_nodes+1));
        for triplet in trip.triplet_iter() {
            let (val, (row, col)) = triplet;
            //assert!(row > col, "upper triangular entry row {} col {}", row, col);
            trip_fixed.add_triplet(row as usize, col as usize, *val);
        }

        let col_format = trip_fixed.to_csc::<i32>();
        println!("is the virus dataset symmetric? {}", sprs::is_symmetric(&col_format));
        println!("virus dataset has {} nonzeros", col_format.nnz());
        return col_format;
    }
    else {
        let col_format = trip.to_csc::<i32>();

        return col_format;
    }
    
    // for i in guy.iter() {
    //     println!("{:?}", i);
    // }
    //println!("density of input {}: {}", filename, guy.density());
}

//make this generic later maybe
pub fn write_mtx(filename: &str, matrix: &CsMat<f64>) {
    sprs::io::write_matrix_market(filename, matrix).ok();
}

// reads a jl_sketch vec of vecs from a file and 
pub fn read_vecs_from_file_flat(filename: &str) -> ffi::FlattenedVec {
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

pub fn write_csv(filename: &str, array: &Array2<f64>) -> Result<(), Box<dyn Error>> {

    // Write the array into the file.
    {
        let file = File::create(filename)?;
        let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        writer.serialize_array2(&array)?;
    }

    // Read an array back from the file
    let file = File::open("data/test.csv")?;
    let mut reader = ReaderBuilder::new().has_headers(false).from_reader(file);
    let array_read: Array2<f64> = reader.deserialize_array2((2, 3))?;

    // Ensure that we got the original array back
    assert_eq!(array_read, array);
    Ok(())
}

//used for testing solver. assumes num_values is equal to number of nodes in graph +1.
//put random values in for all positions except the last, which is set so that the sum
// is equal to 0.

pub fn make_fake_jl_col(num_values: usize) -> ffi::FlattenedVec{
    let mut fake_jl_col: Vec<f64> = vec![0.0; num_values];
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    // add random values for each entry except the last.
    for i in 0..num_values-1 {
        let value = uniform.sample(&mut rng);
        if let Some(position) = fake_jl_col.get_mut(i) {
            *position += value;
        }
        //fake_jl_col.get_mut(i) += value;
    }
    let sum: f64 = fake_jl_col.iter().sum();
    if let Some(position) = fake_jl_col.get_mut(num_values-1){
        *position += -1.0 * sum;
    }
    //fake_jl_col.get_mut(num_values-1) += -1.0 * sum;
    let sum: f64 = fake_jl_col.iter().sum();
    assert!(sum.abs_diff_eq(&0.0, 1e-10), "fake jl sketch vector sum is nonzero: {}", sum);
    let output = ffi::FlattenedVec{vec: fake_jl_col, num_cols: 1, num_rows: num_values};
    return output;

}

//laplacian: &CsMatI<f64, i32>
pub fn create_trivial_rhs(num_values: usize, matrix: &CsMatI<f64,i32>) -> ffi::FlattenedVec {

    println!("hi");
    let indices: Vec<i32> = (0..num_values as i32).collect();
    let mut values: Vec<f64> = vec![0.0; num_values];
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    // add random values for each entry except the last.
    for i in 0..num_values {
        if i%50000 == 0 {
            println!("{}",i);
        }
        let value = uniform.sample(&mut rng);
        if let Some(position) = values.get_mut(i) {
            *position += value;
        }
        //fake_jl_col.get_mut(i) += value;
    }
    println!("done");

    let trivial_solution = CsVecI::<f64, i32>::new(num_values, indices, values);
    println!("vec build done");
    let temp_trivial_rhs = (matrix * &trivial_solution);
    println!("mult done");
    let trivial_rhs = temp_trivial_rhs.to_dense().to_vec();
    println!("conversion done");
    ffi::FlattenedVec{vec: trivial_rhs, num_cols: 1, num_rows: num_values}
}

//pub fn 


