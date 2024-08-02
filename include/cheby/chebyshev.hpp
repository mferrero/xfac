#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

#include <complex>
#include <vector>
#include <functional>
#include <tuple>
#include <cassert>
#include <map>
#include "serialize.h"
#include <iostream>
#include <algorithm>

#define ARMA_DONT_USE_OPENMP
#include <armadillo>

namespace chebyshev
{

    //-------------------------------------------------------- Fast Discrete Cosine Transform ---------------------------------
    // this code aim at calculating cosine transforms dans reverse cosine transform using fft in order to compute cheby weight
    // more about this can be found at https://www.uio.no/studier/emner/matnat/math/nedlagte-emner/MAT-INF2360/v12/siglinbok.pdf p115

    inline arma::vec DCT_(arma::vec const &x)
    // implementation of type II DCT for doubles using arma objects
    {
        uint N = x.size();
        if (N == 1)
            return arma::vec({1.});
        arma::vec x_prepare(N, arma::fill::zeros);
        for (uint k = 0; k <= N / 2 - 1; k++)
            x_prepare.at(k) = x.at(2 * k);
        for (uint k = 0; k <= N / 2 - 1; k++)
            x_prepare.at(N - k - 1) = x.at(2 * k + 1);
        if (N & 1)
            x_prepare.at((N - 1) / 2) = x.at(N - 1);
        arma::cx_vec y = arma::fft(x_prepare);
        arma::vec Y(N);
        for (uint k = 0; k < N; k++)
            Y.at(k) = cos(k * M_PI / (2 * N)) * y.at(k).real() + sin(k * M_PI / (2 * N)) * y.at(k).imag();
        return Y;
    }

    inline arma::cube DCT_matrix(arma::cube const &x)
    // implementation of type II DCT for doubles using arma objects
    {
        uint N = x.n_slices;
        if (N == 1)
            return arma::cube(x.n_rows, x.n_cols, x.n_slices, arma::fill::ones);
        arma::cube x_prepare(x.n_rows, x.n_cols, x.n_slices);
        arma::cube Y(x.n_rows, x.n_cols, x.n_slices);
        for (uint k = 0; k <= N / 2 - 1; k++)
            x_prepare.slice(k) = x.slice(2 * k);
        for (uint k = 0; k <= N / 2 - 1; k++)
            x_prepare.slice(N - k - 1) = x.slice(2 * k + 1);
        if (N & 1)
            x_prepare.slice((N - 1) / 2) = x.slice(N - 1); // even or odd, identical to if(n%2==1)

        // to do the fourier transform of the cube along the slice direction, we first reshape the cube to a matrix with
        // shape x.n_rows * x.n_cols, x.n_slices, we then make an fft of each column vector
        // and then reshape the result back to the original cube form
        arma::mat to_fft = arma::reshape(arma::mat(x_prepare.memptr(), x_prepare.n_elem, 1, false), x.n_rows * x.n_cols, x.n_slices).t();
        arma::cx_mat after_fft = arma::fft(to_fft).st();
        arma::cx_cube y(after_fft.memptr(), x_prepare.n_rows, x_prepare.n_cols, x_prepare.n_slices, false);

        for (uint k = 0; k < N; k++)
            Y.slice(k) = cos(k * M_PI / (2 * N)) * arma::real(y.slice(k)) + sin(k * M_PI / (2 * N)) * arma::imag(y.slice(k));
        return Y;
    }

    inline arma::cx_cube DCT_matrix(arma::cx_cube const &x)
    // implementation of type II DCT for complex using arma objects
    {
        double fac = 2. / x.n_slices;
        arma::cube X_real = DCT_matrix(arma::real(x));
        arma::cube X_imag = DCT_matrix(arma::imag(x));
        return fac * arma::cx_cube(X_real, X_imag);
    }

    inline arma::cx_vec DCT_(arma::cx_vec const &x)
    // implementation of type II DCT for complex using arma objects
    {
        arma::vec X_real = DCT_(arma::real(x));
        arma::vec X_imag = DCT_(arma::imag(x));
        return arma::cx_vec(X_real, X_imag);
    }

    template <typename T>
    inline std::vector<T> DCT(std::vector<T> const &x)
    // implementation of type II DCT
    {
        auto x_conv = arma::conv_to<arma::Col<T>>::from(x);
        auto X = DCT_(x_conv);
        return arma::conv_to<std::vector<T>>::from(X);
    }

    template <typename T>
    inline std::vector<T> DCT_f_to_cheby(std::vector<T> const &x)
    // computation of cheby coef using DCT
    //  evaluate the chebyshev coefficients c from the function evaluated at the chebyshev sampling points (zeros of the chebyshev polynominals)
    {
        auto cheby = DCT(x);
        double fac = 2. / x.size();
        for (uint i = 0; i < x.size(); i++)
        {
            cheby[i] = fac * cheby[i];
        }
        // std::reverse(cheby.begin(), cheby.end());
        return cheby;
    }

    //-------------------------------------------------------- Define some tools first ---------------------------------

    inline std::vector<double> calc_cheby_abscissas(double a, double b, int n)
    {
        // calculate the chebyshev sampling points (zeros of the chebyshev polynominals)
        // x_k = y_k*(b-a)/2 + (b+a)/2, y_k = cos(pi (k + 1/2) / n), k = 0, 1, ..., n-1
        if (a >= b)
            throw std::invalid_argument("calc_cheby_abscissas: a >= b");
        if (n < 1)
            throw std::invalid_argument("calc_cheby_abscissas: n < 1");

        double bpa, bma, y;
        std::vector<double> x(n);
        bma = 0.5 * (b - a);
        bpa = 0.5 * (b + a);
        for (int k = 0; k < n; k++)
        {
            y = cos(M_PI * (k + 0.5) / n);
            x[k] = y * bma + bpa;
        }
        std::reverse(x.begin(), x.end());
        return x;
    }

    //-------------------------------------------------------- The class Chebyshev  ---------------------------------

    template <typename T>
    class Chebyshev
    {
        // Pimped implementation from Numerical Recipes 3. edition.
    public:
        Chebyshev() = default;

        // Construct a Chebyshev interpolant of order n in interval [a, b] from a function, m is the evaluation order
        Chebyshev(std::function<T(double)> func_, double a_, double b_, int n_, int m_ = 0)
            : a{a_}, b{b_}, n{n_}
        {
            if (m_ == 0)
            {
                set_m(n);
            }
            else
            {
                set_m(m_);
            }
            // evaluate the function func_(x) at the chebyshev sampling points (zeros of the chebyshev polynominals)
            auto x = calc_cheby_abscissas(a, b, n);
            std::reverse(x.begin(), x.end());
            std::vector<T> f(n);
            for (int i = 0; i < n; i++)
                f[i] = func_(x[i]);
            c = DCT_f_to_cheby<T>(f); // construct the chebyshev coefficients
            // for (uint i = 0; i < n; i++){
            //     std::cout << "x= " << x[i]  << " f= " << f[i] << " coeff= " << c[i] << "\n";
            // }
            // exit(0);
        }

        // Construct a Chebyshev interpolant in interval [a, b] from the coefficients; m is the evaluation order
        Chebyshev(const std::vector<T> &c_, double a_, double b_, int m_)
            : c{c_}, a{a_}, b{b_}, n{(int)c_.size()}, m{m_}
        {
        }

        // Evaluate the Chebyshev interpolant on point x; m is the evaluation order
        T operator()(double x, int m) const
        {
            // The Chebyshev polynomial is \sum_{k=0}^{m-1} c_k T_k(y) - c_0/2 and it is evaluated
            // at point y = (x - (b+a)/2) / ((b-a)/2)
            double y, y2;
            T d = 0.0, dd = 0.0, sv;
            if ((x - a) * (x - b) > 0.0)
                throw std::invalid_argument("x not in range in Chebyshev call");
            if (m > n || m < 1)
                throw std::invalid_argument("m not correct");

            y = (2.0 * x - a - b) / (b - a); // Change of variable
            y2 = 2.0 * y;

            // use Clenshaw’s recurrence formula
            for (int j = m - 1; j > 0; j--)
            {
                sv = d;
                d = y2 * d - dd + c[j];
                dd = sv;
            }
            return y * d - dd + 0.5 * c[0];
        }

        // Evaluate the Chebyshev interpolant on point x
        T operator()(double x) const
        {
            return (*this)(x, m);
        }

        // Return the Chebyshev interpolant of the function derivative f'(x)
        Chebyshev derivative() const
        {
            double con;
            std::vector<T> cder(n);
            cder[n - 1] = 0.0;
            cder[n - 2] = static_cast<T>(2 * (n - 1)) * c[n - 1];
            for (int j = n - 2; j > 0; j--)
                cder[j - 1] = cder[j + 1] + static_cast<T>(2 * j) * c[j];
            con = 2.0 / (b - a);
            for (int j = 0; j < n; j++)
                cder[j] *= con;
            return Chebyshev(cder, a, b, m);
        }

        // Return the Chebyshev interpolant of the primitive function F(x) = \int_a^x f(t) dt
        Chebyshev integral() const
        {
            double fac = 1.0, con;
            T sum = 0.0;
            std::vector<T> cint(n);
            con = 0.25 * (b - a);
            for (int j = 1; j < n - 1; j++)
            {
                cint[j] = con * (c[j - 1] - c[j + 1]) / (static_cast<double>(j));
                sum += fac * cint[j];
                fac = -fac;
            }
            cint[n - 1] = con * c[n - 2] / ((static_cast<double>(n - 1)));
            sum += fac * cint[n - 1];
            cint[0] = 2.0 * sum;
            return Chebyshev(cint, a, b, m);
        }

        // Set the truncation order
        void set_m(int m_)
        {
            if (m_ > n || m_ < 1)
                throw std::invalid_argument("m not correct");
            m = m_;
        }

        // Get lower bound
        double get_a() const { return a; }

        // Get upper bound
        double get_b() const { return b; }

        void save(std::ostream &out) const
        {
            throw std::runtime_error("Chebyshev: save not implemented");
        }

        static Chebyshev load(std::ifstream &in)
        {
            throw std::runtime_error("Chebyshev: load not implemented");
        }

        void print() const
        {
            std::cout << "Chebyshev: n: " << n << " m: " << m << "\n"
                      << std::flush;
            std::cout << "c=[ ";
            for (auto elem : c)
                std::cout << elem << ", ";
            std::cout << "] \n"
                      << std::flush;
        }

    private:
        std::vector<T> c;
        double a, b;
        int n, m;
        double tol = 1e-14;
    };

    //-------------------------------------------------------- Application helpers --------------------------

    inline std::vector<double> cheby_weights(double x, double a, double b, int n, int deriv = 0, int m = 0)
    {
        // chebyshev weights w_i for integation over full domain:
        //  \int_a^b f(x) dx \approx \sum_i f(x_i) w_i, x_i are the chebyshev abcissas
        auto xi = calc_cheby_abscissas(a, b, n);
        std::vector<double> wi(n);
        for (int i = 0; i < n; i++)
        {
            auto f = [x0 = xi[i]](double x)
            {if (x==x0) return 1.0; else return 0.0; };
            switch (deriv)
            {
            case 0:
                wi[i] = Chebyshev<double>(f, a, b, n, m)(x);
                break;
            case 1:
                wi[i] = Chebyshev<double>(f, a, b, n, m).derivative()(x);
                break;
            case -1:
                wi[i] = Chebyshev<double>(f, a, b, n, m).integral()(x);
                break;
            default:
                throw std::invalid_argument("cheby_weights: case not implemented");
            }
        }
        // std::reverse(wi.begin(), wi.end());
        return wi;
    }

    inline std::vector<double> cheby_multi_xi(std::vector<std::tuple<double, double, size_t>> grid)
    {
        std::vector<double> xi;
        for (auto const &[a, b, n] : grid)
        {

            auto ti = calc_cheby_abscissas(a, b, n);
            xi.insert(std::end(xi), std::begin(ti), std::end(ti));
        }
        return xi;
    }

    inline std::vector<double> cheby_multi_wi(std::vector<std::tuple<double, double, size_t>> grid)
    {
        std::vector<double> wi;
        for (auto const &[a, b, n] : grid)
        {

            auto qi = cheby_weights(b, a, b, n, -1);
            wi.insert(std::end(wi), std::begin(qi), std::end(qi));
        }
        return wi;
    }

    // ---- Multi interval chebyshev

    template <typename T>
    class Cheby
    {
    public:
        Cheby() = default;

        Cheby(std::vector<T> f, std::vector<std::tuple<double, double, size_t>> grid)
        {
            if (grid.size() == 0)
            {
                throw std::invalid_argument("Cheby::init: size is zero");
            }

            for (auto const &[a, b, n] : grid)
            {

                auto x = calc_cheby_abscissas(a, b, n);
                std::vector<T> fi(n);
                for (int i = 0; i < n; i++)
                    fi[i] = f[nges + i];
                std::reverse(fi.begin(), fi.end());
                auto coeffs = DCT_f_to_cheby<T>(fi); // construct the chebyshev coefficients

                av.push_back(a);
                bv.push_back(b);
                nv.push_back(n);
                offsets.push_back(nges);
                nges += n;
                nint += 1;

                cheby.push_back(Chebyshev<T>{coeffs, a, b, static_cast<int>(n)});
            }

            if (!(std::is_sorted(av.begin(), av.end()) || std::is_sorted(av.begin(), av.end())))
            {
                throw std::invalid_argument("Cheby::init: grid not sorted increasingly or intervals");
            }

            if (f.size() != nges)
            {
                throw std::invalid_argument("Cheby::init: f.size() != nges");
            }
        }

        void addInterval(std::vector<T> fi, double a, double b)
        {

            int n = fi.size();

            if (n < 1)
                throw std::invalid_argument("Cheby::addInterval: n < 1");
            if (a >= b)
                throw std::invalid_argument("Cheby::addInterval: a >= b");
            if (std::count(av.begin(), av.end(), a))
                throw std::invalid_argument("Cheby::addInterval: interval bound a already present");
            if (std::count(bv.begin(), bv.end(), b))
                throw std::invalid_argument("Cheby::addInterval: interval bound b already present");

            std::reverse(fi.begin(), fi.end());
            auto coeffs = DCT_f_to_cheby<T>(fi); // construct the chebyshev coefficients

            av.push_back(a);
            bv.push_back(b);
            nv.push_back(n);
            offsets.push_back(nges);
            nges += n;
            nint += 1;

            cheby.push_back(Chebyshev<T>{coeffs, a, b, n});

            if (!(std::is_sorted(av.begin(), av.end()) || std::is_sorted(av.begin(), av.end())))
            {
                throw std::invalid_argument("Cheby::addInterval: grid not sorted increasingly or intervals");
            }
        }

        T eval(double x) const
        {
            size_t ival = interval(x);
            return cheby[ival](x);
        }

        void print() const
        {
            std::cout << "Cheby: number of intervals: " << nint << "\n"
                      << std::flush;
            for (size_t i = 0; i < nint; i++)
            {
                std::cout << "a= " << av[i] << " b= " << bv[i] << " n= " << nv[i] << "\n"
                          << std::flush;
            }
            cheby[0].print();
            exit(0);
        }

    private:
        int interval(double x) const
        {
            for (size_t i = 0; i < nint; i++)
            {
                if (x >= av[i] && x <= bv[i])
                    return i;
            }
            std::string message = "Cheby::interval: x= " + std::to_string(x) + " not in any interval";
            throw std::invalid_argument(message);
        }

        size_t nges = 0;
        size_t nint = 0;
        std::vector<double> av;
        std::vector<double> bv;
        std::vector<int> nv;
        std::vector<int> offsets;
        std::vector<Chebyshev<T>> cheby;
    };

    template <typename T>
    class ChebyMat
    {

    public:
        std::vector<double> xi;
        arma::Cube<T> c;
        int nmin = 20;

        ChebyMat() = default;

        ChebyMat(std::vector<std::tuple<double, double, size_t>> grid)
            : _grid{grid}
        {
            /*
               Construct a piecewise Chebyshev interpolant
               grid is a vector containing an arbitrary number (TODO: for the moment just one) of subintervals in the form of (a, b, n)
               tuples, where [a, b] is the subinterval, on which the functions are discretized on n points.
            */

            if (grid.size() == 0)
                throw std::invalid_argument("ChebyMat::init: size is zero");
            if (grid.size() > 1)
                throw std::invalid_argument("ChebyMat::init: only single grid implemented"); // DEBUG workaround

            std::tie(a, b, n) = grid[0];
            m = n;

            xi = chebyshev::calc_cheby_abscissas(a, b, n);
        }

        void interpolate(std::function<arma::Mat<T>(double)> func)
        {
            auto fi = func(xi[n - 1]);
            arma::Cube<T> fx(fi.n_rows, fi.n_cols, n);
            fx.slice(0) = fi;
            for (int i = 1; i < n; i++)
                fx.slice(i) = func(xi[n - i - 1]);
            c = DCT_matrix(fx);
        }

        // Evaluate the Chebyshev interpolant on point x; m is the evaluation order
        arma::Mat<T> eval(double x) const
        {
            // The Chebyshev polynomial is \sum_{k=0}^{m-1} c_k T_k(y) - c_0/2 and it is evaluated
            // at point y = (x - (b+a)/2) / ((b-a)/2)

            if ((x - a) * (x - b) > 0.0)
                throw std::invalid_argument("x not in range in Chebyshev call");

            arma::Mat<T> d(c.n_rows, c.n_cols, arma::fill::zeros);
            arma::Mat<T> dd(c.n_rows, c.n_cols, arma::fill::zeros);
            arma::Mat<T> sv(c.n_rows, c.n_cols, arma::fill::zeros);

            double y = (2.0 * x - a - b) / (b - a); // Change of variable
            double y2 = 2.0 * y;

            // use Clenshaw’s recurrence formula
            for (int j = m - 1; j > 0; j--)
            {
                sv = d;
                d = y2 * d - dd + c.slice(j);
                dd = sv;
            }
            return y * d - dd + 0.5 * c.slice(0);
        }

        arma::Mat<T> convolute_left(std::function<arma::Mat<T>(double)> func, double x) const
        {
            /* calculate the integral \int_a^x dy func(x - y) * g(y) , where g denotes the chebyshev interpolant */

            auto kernel = [&](double y)
            {arma::Mat<T> res = func(x - y) * eval(y); return res; };

            if (x == a)
            {
                auto tmp = kernel(x);
                return arma::Mat<T>(tmp.n_rows, tmp.n_cols, arma::fill::zeros);
            }
            auto cm = ChebyMat<T>({std::make_tuple(a, x, n)});
            cm.interpolate(kernel);
            cm.integrate();
            return cm.eval(x);
        }

        void convolute_inplace_left(std::function<arma::Mat<T>(double)> func)
        {
            /* replace the chebyshev interpolant g(x) by f(x) = \int_a^x dy func(x - y) * g(y) */
            auto cm = ChebyMat<T>(_grid);
            auto kernel = [&](double x)
            { return convolute_left(func, x); };
            cm.interpolate(kernel);
            c = cm.c;
        }

        void integrate()
        {
            /* replace the chebyshev interpolant g(x) by its primitive f(x) = \int_a^x dy g(y) */
            arma::Cube<T> cint(c.n_rows, c.n_cols, c.n_slices, arma::fill::zeros);
            arma::Mat<T> sum(c.n_rows, c.n_cols, arma::fill::zeros);
            double fac = 1.0;
            double con = 0.25 * (b - a);

            for (int j = 1; j < n - 1; j++)
            {
                cint.slice(j) = con * (c.slice(j - 1) - c.slice(j + 1)) / (static_cast<double>(j));
                sum += fac * cint.slice(j);
                fac = -fac;
            }
            cint.slice(n - 1) = con * c.slice(n - 2) / ((static_cast<double>(n - 1)));
            sum += fac * cint.slice(n - 1);
            cint.slice(0) = 2.0 * sum;
            c = cint;
        }

        int get_n() { return n; }

    private:
        double a;
        double b;
        int n;
        int m;

        std::vector<std::tuple<double, double, size_t>> _grid;
    };

} // end of namespace

#endif // CHEBYSHEV_H
