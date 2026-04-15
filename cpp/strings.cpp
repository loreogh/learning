#include <iostream>
#include <string>

int main() {
    std::string name = "hydrogen";
    std::string element = "H";

    // Concatenation
    std::string label = element + " (" + name + ")";
    std::cout << label << "\n";

    // Size
    std::cout << name.size() << "\n";   // 8

    // Access individual characters
    std::cout << name[0] << "\n";       // 'h'

    // Substring
    std::cout << name.substr(0, 4) << "\n";  // "hydr"

    // Check contents
    if (name == "hydrogen") {
        std::cout << "match\n";
    }

    return 0;
}
