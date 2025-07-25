// this file stores boring functions for the Rust side of the sparsifier implementation. 
// boring means not related to ffi, not really part of the core sparsifier logic, etc.

use sprs::{CsMat};
use sprs::io::{read_matrix_market, write_matrix_market};

pub fn read_mtx(filename: &str) -> CsMat<f64>{
    let trip = read_matrix_market::<f64, usize, &str>(filename).unwrap();
    // for i in trip.triplet_iter() {
    //     println!("{:?}", i);
    // }
    let col_format = trip.to_csc::<usize>();
    // for i in guy.iter() {
    //     println!("{:?}", i);
    // }
    //println!("density of input {}: {}", filename, guy.density());
    return col_format;
}

//make this generic later maybe
pub fn write_mtx(filename: &str, matrix: &CsMat<f64>) {
    sprs::io::write_matrix_market(filename, matrix).ok();
}

//pub fn 