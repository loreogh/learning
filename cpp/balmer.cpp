#include <iostream>
#include <cmath>

int main() {
	
	const double RH = 1.097 * 1e+7;	

	const int nmax = 5;
	for (int n = 3; n <= nmax; n++){
		double dn = n;
		double inv_lambda = RH * (1.0/4 - 1.0/(dn * dn));
		double lambda = 1/inv_lambda;
		std::cout << "n = " << n << ":   " << lambda * 1e+9 << " nm "
					<< "(" << &lambda << ")\n";
	}
		
	return 0;
}
