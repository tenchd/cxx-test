use sprs::{CsMat};

// template types later
pub struct Triplet{
    pub num_nodes: i32, 
    pub col_indices: Vec<i32>,
    pub row_indices: Vec<i32>,
    pub values: Vec<f64>
}
impl Triplet {
    // default constructor just returns three empty vectors.
    pub fn new(num_nodes: i32) -> Triplet {
        let col_indices: Vec<i32> = vec![];
        let row_indices: Vec<i32> = vec![];
        let values: Vec<f64> = vec![];
        Triplet { 
            num_nodes: num_nodes, 
            col_indices: col_indices, 
            row_indices: row_indices, 
            values: values
        }
    }
}

pub struct Sparsifier{
    pub num_nodes: i32,     // number of nodes in input graph. we need to know this at construction time.
    pub new_entries: Triplet,  //stores values that haven't been sparsified yet
    pub current_laplacian: CsMat<f64>,   //stores all values that survive sparsification
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
        let current_laplacian: CsMat<f64> = CsMat::zero((num_nodes.try_into().unwrap(), num_nodes.try_into().unwrap()));     

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
}