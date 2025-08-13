echo 'LOADING INTEL MODULE'
module load intel
echo 'SETTING CXX ENVIRONMENT VARIABLE'
export CXX=/usr/bin/g++
echo 'RUNNING DEFAULT SPARSFIER DEMO'
cargo run