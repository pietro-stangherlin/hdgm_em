//https://github.com/TheAlgorithms/C-Plus-Plus/blob/master/numerical_methods/golden_search_extrema.cpp#L1

/**
 * \file
 * \brief Find extrema of a univariate real function in a given interval using
 * [golden section search
 * algorithm](https://en.wikipedia.org/wiki/Golden-section_search).
 *
 * \see brent_method_extrema.cpp
 * \author [Krishna Vedala](https://github.com/kvedala)
 */
#define _USE_MATH_DEFINES  //< required for MS Visual C++
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>

#define EPSILON 1e-7  ///< solution accuracy limit

/**
 * @brief Get the minima of a function in the given interval. To get the maxima,
 * simply negate the function. The golden ratio used here is:\f[
 * k=\frac{3-\sqrt{5}}{2} \approx 0.381966\ldots\f]
 *
 * @param f function to get minima for
 * @param lim_a lower limit of search window
 * @param lim_b upper limit of search window
 * @return local minima found in the interval
 */
double golden_search_minima(const std::function<double(double)> &f,
                 double lim_a, double lim_b) {
    uint32_t iters = 0;
    double c, d;
    double prev_mean, mean = std::numeric_limits<double>::infinity();

    // golden ratio value
    const double M_GOLDEN_RATIO = (1.f + std::sqrt(5.f)) / 2.f;

    // ensure that lim_a < lim_b
    if (lim_a > lim_b) {
        std::swap(lim_a, lim_b);
    } else if (std::abs(lim_a - lim_b) <= EPSILON) {
        std::cerr << "Search range must be greater than " << EPSILON << "\n";
        return lim_a;
    }

    do {
        prev_mean = mean;

        // compute the section ratio width
        double ratio = (lim_b - lim_a) / M_GOLDEN_RATIO;
        c = lim_b - ratio;  // right-side section start
        d = lim_a + ratio;  // left-side section end

        if (f(c) < f(d)) {
            // select left section
            lim_b = d;
        } else {
            // selct right section
            lim_a = c;
        }

        mean = (lim_a + lim_b) / 2.f;
        iters++;

        // continue till the interval width is greater than sqrt(system epsilon)
    } while (std::abs(lim_a - lim_b) > EPSILON);

    // std::cout << " (iters: " << iters << ") ";
    return prev_mean;
}