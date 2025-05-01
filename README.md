# Shor's Factor

Factor an 8-bit number using Shor's algorithm with Qiskit.

Shor's algorithm is a quantum algorithm designed to efficiently factorize large integers, which is a problem that classical computers struggle to solve quickly. Here's a simplified explanation of how it works:

1. **Problem Setup**: The goal is to factorize a number \( N \) into its prime components. For example, given \( N = 15 \), the algorithm aims to find its factors, \( 3 \) and \( 5 \).

2. **Choose a Random Base**: Select a random number \( a \) such that \( 1 < a < N \) and \( a \) is coprime to \( N \) (i.e., \( \text{gcd}(a, N) = 1 \)).

3. **Period Finding**: The algorithm focuses on finding the period \( r \) of the function \( f(x) = a^x \mod N \). The period \( r \) is the smallest positive integer such that \( a^r \equiv 1 \mod N \).

4. **Quantum Step**: This is where the quantum computer shines. A quantum circuit is used to create a superposition of states and apply the Quantum Fourier Transform (QFT) to extract the period \( r \). This step is exponentially faster than classical methods.

5. **Classical Post-Processing**: Once \( r \) is found, the algorithm checks if \( r \) is even and if \( a^{r/2} \pm 1 \) are non-trivial factors of \( N \). If these conditions are met, the factors are computed as \( \text{gcd}(a^{r/2} - 1, N) \) and \( \text{gcd}(a^{r/2} + 1, N) \).

6. **Repeat if Necessary**: If the conditions aren't met, the algorithm repeats with a different random base \( a \).

The quantum advantage lies in the period-finding step, which is exponentially faster than classical approaches. This efficiency makes Shor's algorithm a potential threat to classical cryptographic systems like RSA, which rely on the difficulty of factoring large numbers.
