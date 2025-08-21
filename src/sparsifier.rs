use sprs::{CsMatI, CsMatBase, TriMatBase, TriMatI};
use std::ops::Add;
use rand::Rng;
use approx::AbsDiffEq;

use crate::utils::{create_trivial_rhs, make_fake_jl_col};
use crate::ffi;

// template types later
#[derive(Clone)]
pub struct Triplet{
    pub num_nodes: i32, 
    pub col_indices: Vec<i32>,
    pub row_indices: Vec<i32>,
    pub diagonal: Vec<f64>,
    pub values: Vec<f64>
}
impl Triplet {
    // default constructor that just returns three empty vectors.
    pub fn new(num_nodes: i32) -> Triplet {
        let col_indices: Vec<i32> = vec![];
        let row_indices: Vec<i32> = vec![];
        let diagonal: Vec<f64> = vec![0.0; num_nodes.try_into().unwrap()];
        let values: Vec<f64> = vec![];
        Triplet { 
            num_nodes: num_nodes, 
            col_indices: col_indices, 
            row_indices: row_indices, 
            diagonal: diagonal,
            values: values
        }
    }

    // inserts an edge with a weight/value into the triplet format. keeps track of diagonal too.
    pub fn insert(&mut self, v1: i32, v2: i32, value: f64) {
        // make sure node IDs are valid (0 indexed)
        assert!(v1 < self.num_nodes);
        assert!(v2 < self.num_nodes);

        //input files may have diagonals; we should skip them
        // this is a kludge for reading in mtx files. later this logic should be moved elsewhere.
        if (v1 != v2) {
            // insert -value into v1,v2
            self.row_indices.push(v1);
            self.col_indices.push(v2);
            self.values.push(value*-1.0);

            // insert -value into v2,v1
            self.row_indices.push(v2);
            self.col_indices.push(v1);
            self.values.push(value*-1.0);

            // add 1 to diagonal entries v1,v1 and v2,v2
            self.diagonal[<i32 as TryInto<usize>>::try_into(v1).unwrap()] += value;
            self.diagonal[<i32 as TryInto<usize>>::try_into(v2).unwrap()] += value;
        }
    }

    pub fn process_diagonal(&mut self) {
        // add diagonal entries to triplet format
        for (index, value) in self.diagonal.iter().enumerate() {
            let new_index = index as i32;
            self.row_indices.push(new_index);
            self.col_indices.push(new_index);
            self.values.push(*value);
        }
    }

    pub fn to_csc(self) -> CsMatI::<f64, i32> {

        //I think i need to change this so that there's an empty row and column at the end. i should be able to do this by simply setting the number of rows and cols
        // equal to self.num_nodes+1. since this is a triplet rep i think that's ok because neither the row indices or col indices rely on the number of rows or cols.
        // (except for overflow purposes.)
        let trip_form: TriMatBase<Vec<i32>, Vec<f64>>  = TriMatI::<f64, i32>::from_triplets((self.num_nodes as usize, self.num_nodes as usize), self.row_indices, self.col_indices, self.values);
        let csc_form: CsMatBase<f64, i32, Vec<i32>, Vec<i32>, Vec<f64>, _> = trip_form.to_csc();

        csc_form
    }

    pub fn delete_state(&mut self) {

        self.col_indices = vec![];
        self.row_indices = vec![];
        self.diagonal = vec![0.0; self.num_nodes.try_into().unwrap()];
        self.values = vec![];

    }

    pub fn display(&self) {
        println!("triplet values:");
        for value in &self.col_indices {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.row_indices {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.values {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.diagonal {
            print!("{}, ", value);
        }
        println!("");
    }
}

pub struct Sparsifier{
    pub num_nodes: i32,     // number of nodes in input graph. we need to know this at construction time.
    pub new_entries: Triplet,  //stores values that haven't been sparsified yet
    pub current_laplacian: CsMatBase<f64, i32, Vec<i32>, Vec<i32>, Vec<f64>, i32>,   //stores all values that survive sparsification
    pub threshold: i32,     //sparsify if size(new_entries) + size(current_laplacian) > threshold. computed from num_nodes, epsilon, and constants
                              // set to be row_constant*beta*nodesize in line 3(b) of alg pseudocode
    pub epsilon: f64,     //epsilon controls space (aggressivenes of sampling) and approximation factor guarantee.
    pub beta_constant: i32,    // set to be 200 in line 1 of alg pseudocode, probably can be far smaller
    pub row_constant: i32,    // set to be 20 in line 3(b) of alg pseudocode, probably can be far smaller
    pub beta: i32,     // parameter defined in line 1 of alg pseudocode
    pub verbose: bool,    //when true, prints a bunch of debugging info
}

impl Sparsifier {
    pub fn new(num_nodes: i32, epsilon: f64, beta_constant: i32, row_constant: i32, verbose: bool) -> Sparsifier {
        // as per line 1
        let beta = (epsilon.powf(-2.0) * (beta_constant as f64) * (num_nodes as f64).log(2.0)).round() as i32;
        // as per 3(b) condition
        let threshold = num_nodes * beta * row_constant;
        // initialize empty new elements triplet vectors
        let new_entries = Triplet::new(num_nodes);
        // initialize empty sparse matrix for the laplacian
        // it needs to have num_nodes+1 rows and cols actually
        let current_laplacian: CsMatBase<f64, i32, Vec<i32>, Vec<i32>, Vec<f64>, _> = CsMatI::<f64, i32>::zero((num_nodes.try_into().unwrap(), num_nodes.try_into().unwrap()));     

        if verbose {println!("brother you just built a sparsifier");}
        

        Sparsifier{
            num_nodes: num_nodes,
            new_entries: new_entries,
            current_laplacian: current_laplacian,
            threshold: threshold,
            epsilon: epsilon,
            beta_constant: beta_constant,
            row_constant: row_constant,
            beta: beta,
            verbose: verbose,
        }
    }

    // returns # of edges in the entire sparsifier (including the new edges in triplet form and old edges in sparse matrix form)
    // note that currently it overcounts for laplacian since it also counts diagonals. maybe change this later?
    pub fn size_of(&self) -> i32 {
        <usize as TryInto<i32>>::try_into(self.new_entries.col_indices.len()).unwrap()  // total entries in triplet form
        + 
        <usize as TryInto<i32>>::try_into(self.current_laplacian.nnz()).unwrap()  // total nonzeros in laplacian
        
    }

    // inserts an edge into the sparsifier. if this makes the size of the sparsifier cross the threshold, trigger sparsification.
    pub fn insert(&mut self, v1: i32, v2: i32, value: f64) {
        // insert -1 into v1,v2 and v2,v1. add 1 to v1,v1 and v2,v2
        // problem: duplicate values in diagonals for triplets. 
        // am i assuming that each edge appears at most once in the stream? if that's violated, i could have duplicate entries in the triplets
        
        // make sure node IDs are valid (0 indexed)
        assert!(v1 < self.num_nodes);
        assert!(v2 < self.num_nodes);

        // ignore diagonal entries, upper triangular entries, and entries with 0 value
        if (v1 > v2 && value != 0.0){
            self.new_entries.insert(v1, v2, value);
            //println!("inserting ({}, {}) with value {}", v1, v2, value);
        }

        //TODO: if it's too big, trigger sparsification step
    }

    // returns probabilities for all nonzero entries in laplacian. placeholder for now
    pub fn get_probs(length: usize) -> Vec<f64> {
        // need to subsample, but only off-diagonals.
        vec![0.5; length]
    }

    pub fn flip_coins(length: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let coins = vec![rng.gen_range(0.0..1.0); length];
        coins
    }

    pub fn sparsify(&mut self, end_early: bool) {
        // this is dummy sparsifier code until i integrate it with the c++ code
        // apply diagonals to new triplet entries
        self.new_entries.process_diagonal();
        // get the new entries in csc format
        // improve this later; currently it clones the triplet object which uses extra memory
        let new_stuff = self.new_entries.clone().to_csc();
        // clear the new entries from the triplet representation
        self.new_entries.delete_state();
        // add the new entries to the laplacian
        self.current_laplacian = self.current_laplacian.add(&new_stuff);
        println!("checking diagonal after populating laplacian:");
        self.check_diagonal();

        if end_early {
            return;
        }

        // ------ TRYING A SOLVE STEP HERE TO DEBUG --
        println!("is the matrix symmetric? {}", sprs::is_symmetric(&self.current_laplacian));

        //create a trivial solution via forward multiplication. for testing purposes, will remove later
        //NOTE: currently this call takes a LONG time. like 10-20 minutes. DIAGNOSE
        let trivial_right_hand_side = create_trivial_rhs(self.num_nodes as usize, &self.current_laplacian);

        println!("done generating trivial rhs");
        let col_ptrs: Vec<i32> = self.current_laplacian.indptr().as_slice().unwrap().to_vec();
        let row_indices: Vec<i32> = self.current_laplacian.indices().to_vec();
        let values: Vec<f64> = self.current_laplacian.data().to_vec();
        println!("jl sketch col has {} entries. lap has {} cols and {} nzs", trivial_right_hand_side.vec.len(), col_ptrs.len()-1, row_indices.len());
        println!("there are {} nonzeros in the last column", col_ptrs[self.num_nodes as usize] - col_ptrs[(self.num_nodes-1) as usize]);
        let dummy = ffi::run_solve_lap(trivial_right_hand_side, col_ptrs, row_indices, values, self.num_nodes);
        // ------ END DEBUGGING SOLVE STEP -----------

        let diag = self.current_laplacian.diag();
        let diag_nnz = diag.nnz();
        let num_nnz = (self.current_laplacian.nnz() - diag_nnz) / 2;
        // janky check for integer rounding. i hang my head in shame
        assert_eq!(num_nnz*2, (self.current_laplacian.nnz()-diag_nnz));
        // get probabilities for each edge
        let probs = Self::get_probs(num_nnz);
        let coins = Self::flip_coins(num_nnz);
        //encodes whether each edge survives sampling or not. True means it is sampled, False means it's not sampled
        let outcomes: Vec<bool> =  probs.clone().into_iter().zip(coins.into_iter()).map(|(p, c)| c < p).collect();

        let mut counter = 0;
        for (value, (row, col)) in self.current_laplacian.iter() {
            let outcome = outcomes[counter];
            let prob = probs[counter];
            // still need to update entries in off-diagonal and diagonal. including reweighting. probably put this in a function
            // just realized that updating both symmetric versions of elements forces row-order accesses in the upper diagonal. should optimize this eventually
        }

        println!("checking diagonal after sampling");
        self.check_diagonal();

    }

    pub fn sparse_display(&self) {
        println!("laplacian: ");
        for (value, (row, col)) in self.current_laplacian.iter() {
            print!("({}, {}) has value {} ", row, col, value);
        }
        println!("");
    }

    // see if there's a library function to make summing cols more efficient later.
    // also find a way to write efficient code to sum rows. for now, maybe just do em as you do cols?
    pub fn check_diagonal(&self) {
        for (col_index, col_vector) in self.current_laplacian.outer_iterator().enumerate() {
            let mut sum:f64 = 0.0;
            for (_, value) in col_vector.iter() {
                sum += value;
            }
            // check each column is sum 0. floating point error means you'll be a little off
            assert!(sum.abs_diff_eq(&0.0, 1e-10), "column where we messed up: {}. column sum: {}.", col_index, sum);
        }
        println!("each column sums to 0. diagonal check passed.");
    }
}