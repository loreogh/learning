#include <iostream>
#include <array>
#include <cmath>

double norm(const std::array<double, 3> &v){
	return std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

void normalize(std::array<double, 3> &v){
	double n = norm(v);
	v[0] /= n;
	v[1] /= n;
	v[2] /= n;
}

int main() {
	
	std::array<double, 3> vec = {3.0, 1.0, 2.0};
	
	std::cout << "norm before normalization: " << norm(vec) << "\n";
	
	normalize(vec);
	std::cout << "norm after normalization: " << norm(vec) << "\n";

	return 0;
}
