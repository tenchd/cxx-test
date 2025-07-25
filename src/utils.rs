// this file stores boring functions for the Rust side of the sparsifier implementation. 
// boring means not related to ffi, not really part of the core sparsifier logic, etc.

use sprs::{CsMat};
use sprs::io::{read_matrix_market, write_matrix_market};

use crate::sparsifier::{Sparsifier,Triplet};

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

pub struct InputStream {
    pub input_matrix: CsMat<f64>,
//    pub input_iterator: 
    pub num_nodes: usize,
//    pub num_edges: usize,
}

impl InputStream {
    // deal with diagonals?
    // if the graph is symmetric, de-symmetrize it ideally
    // how does the mtx reader handle symmetry?
    pub fn new(filename: &str) -> InputStream {
        let mut input = read_mtx(filename);
        // zeroed diagonal entries remain explicitly represented using this format.
        // fix this later.
        let num_nodes = input.outer_dims();
        assert_eq!(input.outer_dims(), input.inner_dims());
        for result in input.diag_iter_mut() {
            match result {
                Some(x) => *x = 0.0,
                None => println!("problem iterating over diagonal"),
            }
        }
        // for value in input.iter() {
        //     println!("{:?}", value);
        // }
        InputStream{
            input_matrix: input,
            num_nodes: num_nodes,
        }
    }

    pub fn run_stream(&self, epsilon: f64, beta_constant: i32, row_constant: i32, verbose: bool) {
        let mut sparsifier = Sparsifier::new(self.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose);

        for (value, (row, col)) in self.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        sparsifier.new_entries.display();
        //s.sparse_display();
        sparsifier.sparsify();
        //s.new_entries.display();
        sparsifier.sparse_display();

        sparsifier.check_diagonal();

    }
}

//pub fn 