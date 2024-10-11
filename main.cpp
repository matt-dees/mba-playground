#include <stdint.h>
#include <random>
#include <iostream>

uint64_t P(uint64_t x) {
    // P(x) = 2^33 * 179 x^2 + (2^34 + 1)x + 1221118466;
    const uint64_t a = 1537598291968ULL;
    const uint64_t b = 17179869185ULL;
    const uint64_t c = 1221118466ULL;
    return a * x * x + b * x + c;
}

uint64_t Q(uint64_t x) {
    // Q(x) = P^-1(x) = 18446742536111259648 x ^ 2 + 10490288244149714945 x + 6494527593041573374
    const uint64_t a = 18446742536111259648ULL;
    const uint64_t b = 10490288244149714945ULL;
    const uint64_t c = 6494527593041573374ULL;
    return a * x * x + b * x + c;
}

uint64_t Q_P(uint64_t x) {
    // x*y = (x&y)*(x|y) + (x&~y)*(~x&y)
    uint64_t r = 
    1221118466ULL 
    + 17179869185UL * (6494527593041573374UL + (10490288244149714945UL & x)*(10490288244149714945UL | x) + (10490288244149714945UL & ~x) * (~10490288244149714945UL & x) + 18446742536111259648UL * x * x) 
    + 1537598291968UL * (6494527593041573374UL + 10490288244149714945UL * x + 18446742536111259648UL * x * x) * (6494527593041573374UL + 10490288244149714945UL * x + 18446742536111259648UL * x * x);
    return r;
}

int main() {
    std::random_device rd;

    std::mt19937_64 e2(rd());

    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFF);
    for (uint64_t i = 0; i < 100; i++) {
        uint64_t x = dist(e2);
        std::cout << "x: \t" << x << "\t --> \t" << Q_P(x) << std::endl;
    }
}