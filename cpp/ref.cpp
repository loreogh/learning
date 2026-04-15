#include <iostream>

int main() {
    double x = 3.14;
    double& ref = x;    // ref is an alias for x

    ref = 2.71;         // modifies x through the alias
    std::cout << x << "\n";   // prints 2.71

    std::cout << &x   << "\n";  // same address
    std::cout << &ref << "\n";  // same address — they ARE the same variable
    return 0;
}
