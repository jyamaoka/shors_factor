from qiskit.algorithms import Shor
from qiskit.primitives import Sampler
from qiskit.utils import algorithm_globals
from typing import Optional

def factor_number(n: int, seed: Optional[int] = None) -> Optional[int]:
    """
    Factor an 8-bit integer using Shor's algorithm.

    Args:
        n (int): Number to factor (must be < 256).
        seed (Optional[int]): Seed for reproducibility.

    Returns:
        Optional[int]: A non-trivial factor of n, or None if failed.
    """
    if n < 3 or n >= 256:
        raise ValueError("Only 8-bit integers (3 <= n < 256) are supported.")

    if seed is not None:
        algorithm_globals.random_seed = seed

    shor = Shor()
    result = shor.factor(N=n)
    return result.factors[0][0] if result.factors else None

from qiskit import QuantumCircuit, Aer, execute
import numpy as np
from math import gcd

def factorize(N):
    if N <= 1 or N >= 256:
        raise ValueError("Please provide an 8-bit number greater than 1 and less than 256.")

    def get_random_base(N):
        base = np.random.randint(2, N)
        while gcd(base, N) != 1:
            base = np.random.randint(2, N)
        return base

    base = get_random_base(N)
    print(f"Random base selected: {base}")

    # Quantum circuit for period finding
    circuit = QuantumCircuit(4, 4)
    circuit.h(range(4))  # Apply Hadamard gates
    # Add the implementation for modular exponentiation and measurement...

    simulator = Aer.get_backend('qasm_simulator')
    result = execute(circuit, simulator).result()

    # Extract period and compute factors (simplified here)
    factors = [gcd(base**period - 1, N), gcd(base**period + 1, N)]
    return factors

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import QFT
from math import gcd
from fractions import Fraction
import numpy as np

def qpe_modexp(a: int, N: int, n_count: int) -> QuantumCircuit:
    """Quantum phase estimation to find the period of a^x mod N."""
    qc = QuantumCircuit(n_count + 4, n_count)

    # Initialize counting qubits in |+⟩ state
    for q in range(n_count):
        qc.h(q)

    # Initialize target in |1⟩
    qc.x(n_count)

    # Apply controlled modular multipliers of a^{2^j}
    for q in range(n_count):
        qc = apply_c_amod15(qc, a**(2**q) % N, q, n_count)

    # Apply inverse QFT
    qc.append(QFT(num_qubits=n_count, inverse=True, do_swaps=True), range(n_count))

    # Measure
    qc.measure(range(n_count), range(n_count))
    return qc

def apply_c_amod15(qc, a: int, control_qubit: int, n_count: int) -> QuantumCircuit:
    """Controlled modular multiplication by a mod 15."""
    x = n_count  # first target qubit
    if a == 2:
        qc.cswap(control_qubit, x, x+1)
        qc.cswap(control_qubit, x+1, x+2)
        qc.cswap(control_qubit, x+2, x+3)
    elif a == 4:
        qc.cswap(control_qubit, x, x+2)
        qc.cswap(control_qubit, x+1, x+3)
    elif a == 7:
        qc.cswap(control_qubit, x, x+3)
        qc.cswap(control_qubit, x+1, x+2)
    elif a == 8:
        qc.cswap(control_qubit, x, x+2)
        qc.cswap(control_qubit, x+1, x+3)
        qc.cswap(control_qubit, x+2, x+3)
    elif a == 11:
        qc.cswap(control_qubit, x, x+1)
        qc.cswap(control_qubit, x+1, x+2)
        qc.cswap(control_qubit, x+2, x+3)
    elif a == 13:
        qc.cswap(control_qubit, x, x+2)
        qc.cswap(control_qubit, x+1, x+3)
        qc.cswap(control_qubit, x+2, x+3)
    return qc

def get_period(phase: int, n_count: int) -> int:
    """Convert measured phase to period r."""
    decimal = phase / (2 ** n_count)
    frac = Fraction(decimal).limit_denominator(15)
    return frac.denominator

def shor_manual(N: int, a: int = 7) -> int:
    """Attempt to factor N manually with Qiskit."""
    if gcd(a, N) != 1:
        return gcd(a, N)

    n_count = 8  # number of counting qubits
    qc = qpe_modexp(a, N, n_count)

    # Execute on QASM simulator
    backend = Aer.get_backend('qasm_simulator')
    result = execute(qc, backend, shots=1).result()
    counts = result.get_counts()
    phase_bin = max(counts, key=counts.get)
    phase_int = int(phase_bin, 2)

    # Get the period
    r = get_period(phase_int, n_count)
    if r % 2 != 0:
        return None

    # Compute factors
    plus = pow(a, r // 2) + 1
    minus = pow(a, r // 2) - 1
    factor1 = gcd(plus, N)
    factor2 = gcd(minus, N)

    if factor1 == 1 or factor1 == N:
        return None
    return factor1

