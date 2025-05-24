#pragma once
#include <functional>

double golden_search_minima(const std::function<double(double)> &f,
                 double lim_a, double lim_b);