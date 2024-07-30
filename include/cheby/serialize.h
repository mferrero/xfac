#pragma once

#include <fstream>
#include <iomanip>
#include <functional>
#include <typeinfo>
#define ARMA_DONT_USE_OPENMP
#include <armadillo>

using uint = unsigned int;

namespace xfac {

template< typename T >
void write_mat(const T & mat, std::ostream& out){
    out << mat.n_rows << " " << mat.n_cols << "\n";
    out << typeid(T).name() << "\n";
    for (uint i=0; i < mat.n_rows; i++){
        for (uint j=0; j < mat.n_cols; j++)
           out  << std::setprecision(18) << mat(i, j) << " ";
        out << "\n";
    }
    out << "\n";
}

template< typename S, typename T >
void write_cube(const S & cube, std::ostream& out){
    out << cube.n_rows << " " << cube.n_cols << " " << cube.n_slices << "\n";
    out << typeid(T).name() << "\n";
    for (uint i=0; i < cube.n_slices; i++)
        write_mat<T>(cube.slice(i), out);
}

template< typename S, typename T >
void write_sequence(const S & sequence, std::ostream& out, std::function<void(const T&, std::ostream&)> wfunc){
    out << sequence.size() << "\n";
    out << typeid(T).name() << "\n";
    for (auto const& x:sequence)
        wfunc(x, out);
    out << "\n";
}

template< typename S, typename T >
void write_map(const std::map<S, T> & sequence, std::ostream& out, std::function<void(const S&, std::ostream&)> wkey , std::function<void(const T&, std::ostream&)> wval){
    out << sequence.size() << "\n";
    out << typeid(T).name() << "\n";
    for (auto const& [key, val] : sequence){
        wkey(key, out);
        wval(val, out);
    }
    out << "\n";
}

template< typename S, typename T >
S read_sequence(std::ifstream& in, std::function<T(std::ifstream&)> rfunc){
    int size;
    in >> size;
    std::string data_type;
    in >> data_type;
    if(data_type != typeid(T).name()) throw std::runtime_error("read_sequence(): data_type != T");
    S sequence;
    for (auto i=0; i < size; i++){
        auto elem = rfunc(in);
        sequence.push_back(elem);
    }
    return sequence;
}

template< typename T >
T read_mat(std::ifstream& in){
    int n_rows, n_cols;
    in >> n_rows;
    in >> n_cols;
    std::string data_type;
    in >> data_type;
    if(data_type != typeid(T).name()) throw std::runtime_error("read_mat(): data_type != T");
    T mat(n_rows, n_cols);
    for (auto i=0; i < n_rows; i++){
        for (auto j=0; j < n_cols; j++)
           in >> mat(i, j);
    }
    return mat;
}

template< typename S, typename T >
S read_cube(std::ifstream& in){
    int n_rows, n_cols, n_slices;
    in >> n_rows;
    in >> n_cols;
    in >> n_slices;
    std::string data_type;
    in >> data_type;
    if(data_type != typeid(T).name()) throw std::runtime_error("read_cube(): data_type != T");
    S cube(n_rows, n_cols, n_slices);
    for (auto i=0; i < n_slices; i++)
        cube.slice(i) = read_mat<T>(in);
    return cube;
}

template< typename S, typename T >
std::map<S, T> read_map(std::ifstream& in, std::function<S(std::ifstream&)> rkey , std::function<T(std::ifstream&)> rval){
    std::map<S, T> a;
    int size;
    in >> size;
    std::string data_type;
    in >> data_type;
    if(data_type != typeid(T).name()) throw std::runtime_error("read_map(): data_type != T");
    for (int i=0; i < size; i++){
        auto key = rkey(in);
        auto val = rval(in);
        a[key] = val;
    }
    return a;
}

template< typename T >
inline void saveScalar(const T& x, std::ostream& out){out << x << " ";};

template< typename T >
inline T readScalar(std::ifstream& in){T x; in >> x; return x;};

template< typename T >
inline void saveNumeric(const T& x, std::ostream& out){out << std::setprecision(18) << x << " ";};

// -- some specializations

inline void saveCube(const arma::cx_cube& x, std::ostream& out){write_cube<arma::cx_cube, arma::cx_mat>(x, out);};
inline void saveMat(const arma::cx_mat& x, std::ostream& out){write_mat<arma::cx_mat>(x, out);};

inline arma::cx_cube readCube(std::ifstream& in){return read_cube<arma::cx_cube, arma::cx_mat>(in);};
inline arma::cx_mat readMat(std::ifstream& in){return read_mat<arma::cx_mat>(in);};




} // end namespace xfac

