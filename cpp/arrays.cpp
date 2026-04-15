#include <iostream>
#include <array>

int main() {
	
	std::array<double, 3> position = {1.0, 0.5, -2.0};
	std::array<double, 3> momentum = {0.0, 1.0, 0.0};

	std::cout << "first element : " << position[0]     << "\n";
	std::cout << "size          : " << position.size() << "\n";  // 3

	// Bounds-checked access (throws an exception if out of range)
	std::cout << "bounds check  : " << position.at(1)  << "\n";

	return 0;

}
