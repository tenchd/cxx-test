use sprs::{CsMat,CsMatI,TriMat,TriMatI,CsVec,CsVecI};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};



pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn make_random_matrix(num_rows: usize, nnz: usize, csc: bool) -> CsMat<f64> {
    let mut trip: TriMat<f64> = TriMat::new((num_rows, num_rows));
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    for _ in 0..nnz {
        let row_pos = rng.gen_range(0..num_rows);
        let col_pos = rng.gen_range(0..num_rows);
        let value = uniform.sample(&mut rng);
        trip.add_triplet(row_pos, col_pos, value);
    }
    if csc {
        return trip.to_csc();
    }
    trip.to_csr()
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
    use test::Bencher;

    //For benchmarking how long a sparse matrix x dense vector multiplication takes.
    pub fn spmv_basic(num_rows: usize, nnz: usize, csc: bool, b: &mut Bencher) {
        // let mut mat_type = "";
        // if csc {
        //     mat_type = "CSC";
        // }
        // else {
        //     mat_type = "CSR";
        // }
        // println!("Testing SPMV time for a {} x {} matrix in {} form with {} nonzeros", num_rows, num_rows, mat_type, nnz);
        let mat = make_random_matrix(num_rows, nnz, csc);
        let vector = make_random_vec(num_rows);
        //let result = &mat * &vector;
        b.iter(|| &mat * &vector);
        //assert!(result.nnz()>0);
    }

    //benchmark a small multiplication when the matrix is in csc form
    #[bench]
    fn spmv1c(b: &mut Bencher){
        spmv_basic(10,20,true, b);
    }       

    //benchmark a small multiplication when the matrix is in csr form
    #[bench]
    fn spmv1r(b: &mut Bencher){
        spmv_basic(10,20,false, b);
    }
    
    #[bench]
    fn spmv2c(b: &mut Bencher){
        spmv_basic(100,2000,true, b);
    }

    #[bench]
    fn spmv2r(b: &mut Bencher){
        spmv_basic(100,2000,false, b);
    }

    #[bench]
    fn spmv3c(b: &mut Bencher){
        spmv_basic(1000,200000,true, b);
    }

    #[bench]
    fn spmv3r(b: &mut Bencher){
        spmv_basic(1000,200000,false, b);
    }

    #[bench]
    fn spmv4c(b: &mut Bencher){
        spmv_basic(10000,2000000,true, b);
    }

    #[bench]
    fn spmv4r(b: &mut Bencher){
        spmv_basic(10000,2000000,false, b);
    }


}