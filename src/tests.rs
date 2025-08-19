use sprs::{CsMat,CsMatI,TriMat,TriMatI,CsVec,CsVecI};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn make_random_matrix(num_rows: usize, nnz: usize) -> CsMat<f64> {
    let mut trip: TriMat<f64> = TriMat::new((num_rows, num_rows));
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    for _ in 0..nnz {
        let row_pos = rng.gen_range(0..num_rows);
        let col_pos = rng.gen_range(0..num_rows);
        let value = uniform.sample(&mut rng);
        trip.add_triplet(row_pos, col_pos, value);
    }
    trip.to_csc()
}

pub fn make_random_vec(num_values: usize) -> CsVec<f64> {

    let indices: Vec<usize> = (0..num_values).collect();
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

    let rand_vec = CsVec::new(num_values, indices, values);
    rand_vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spmv_basic() {
        let num_rows = 10;
        let nnz = 20;
        let mat = make_random_matrix(num_rows, nnz);
        let vector = make_random_vec(num_rows);
        let result = &mat * &vector;
        assert!(result.nnz()>0);
    }
}