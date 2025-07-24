use sprs::{CsMatI, CsMatBase, TriMatBase, TriMatI};
use std::ops::Add;

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

        // insert -1 into v1,v2
        self.row_indices.push(v1);
        self.col_indices.push(v2);
        self.values.push(value);

        // insert -1 into v2,v1
        self.row_indices.push(v2);
        self.col_indices.push(v1);
        self.values.push(value);

        // add 1 to diagonal entries v1,v1 and v2,v2
        self.diagonal[<i32 as TryInto<usize>>::try_into(v1).unwrap()] += value;
        self.diagonal[<i32 as TryInto<usize>>::try_into(v2).unwrap()] += value;
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
    pub fn insert(&mut self, v1: i32, v2: i32) {
        // insert -1 into v1,v2 and v2,v1. add 1 to v1,v1 and v2,v2
        // problem: duplicate values in diagonals for triplets. 
        // am i assuming that each edge appears at most once in the stream? if that's violated, i could have duplicate entries in the triplets
        
        // make sure node IDs are valid (0 indexed)
        assert!(v1 < self.num_nodes);
        assert!(v2 < self.num_nodes);

        // insert -1 into v1,v2
        self.new_entries.row_indices.push(v1);
        self.new_entries.col_indices.push(v2);
        self.new_entries.values.push(-1.0);

        // insert -1 into v2,v1
        self.new_entries.row_indices.push(v2);
        self.new_entries.col_indices.push(v1);
        self.new_entries.values.push(-1.0);

        // add 1 to diagonal entries v1,v1 and v2,v2
        self.new_entries.diagonal[<i32 as TryInto<usize>>::try_into(v1).unwrap()] += 1.0;
        self.new_entries.diagonal[<i32 as TryInto<usize>>::try_into(v2).unwrap()] += 1.0;

        //TODO: if it's too big, trigger sparsification step
    }

    // returns probabilities for all nonzero entries in laplacian.
    pub fn get_probs(&self) -> Vec<f64> {
        // need to subsample, but only off-diagonals.
        vec![]
    }

    pub fn sparsify(&mut self) {
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


    }

    pub fn sparse_display(&self) {
        println!("laplacian: ");
        for (value, (row, col)) in self.current_laplacian.iter() {
            print!("({}, {}) has value {} ", row, col, value);
        }
        println!("");
    }
}