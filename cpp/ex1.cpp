#include <iostream>
#include <cmath>

int main() {
	double E_GS = 13.6;
	double n = 3.0;
	double E_excited = E_GS / (n * n);

	std::cout << "Energy at n=3 is " << E_excited << " eV\n";
	std::cout << "Square root of 2 is " << std::sqrt(2.0) << "\n";

	return 0;
}
