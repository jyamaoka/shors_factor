from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.synthesis.qft import synth_qft_full as QFT
#from qiskit.circuit.library import QFTGate as QFT
from math import gcd, ceil, log
from fractions import Fraction
import numpy as np
from typing import Optional, List
from shors_factor.utils import is_prime, check_if_power, get_random_base
import warnings

def qpe_modexp(a: int, N: int, nbits: int) -> QuantumCircuit:
    """Quantum phase estimation to find the period of a^x mod N."""
    qc = QuantumCircuit(nbits + 4, nbits)

    # Initialize counting qubits in |+⟩ state
    for q in range(nbits):
        qc.h(q)

    # Initialize target in |1⟩
    qc.x(nbits)

    # Apply controlled modular multipliers of a^{2^j}
    for q in range(nbits):
        qc = apply_c_amod15(qc, a**(2**q) % N, q, nbits)

    # Apply inverse QFT
    qc.append(QFT(num_qubits=nbits, inverse=True, do_swaps=True), range(nbits))

    # Measure
    qc.measure(range(nbits), range(nbits))
    return qc

def apply_c_amod15(qc, a: int, control_qubit: int, nbits: int) -> QuantumCircuit:
    """Controlled modular multiplication by a mod 15."""
    x = nbits  # first target qubit
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

def get_period(phase: int, nbits: int) -> int:
    """Convert measured phase to period r."""
    decimal = phase / (2 ** nbits)
    frac = Fraction(decimal).limit_denominator(15)
    return frac.denominator

def run_qpe(a: int, N: int) -> Optional[List[int]]:
    """Run quantum phase estimation to find the period of a^x mod N."""
    # Check if N is 8 bit
    if N < 3 or N >= 256:
        raise ValueError("Only 8-bit integers (3 <= n < 256) are supported.")

    # get right number of bits
    nbits = N.bit_length()

    # get circuit    
    qc = qpe_modexp(a, N, nbits)

    # Create QASM simulator
    machine = AerSimulator() # could be real hardware

    # Run the circuit
    compiled_circuit = transpile(qc, machine)
    job = machine.run(compiled_circuit)
    
    # Get the result and process it
    result = job.result()
    counts = result.get_counts(qc)
    phase_bin = max(counts, key=counts.get)
    phase_int = int(phase_bin, 2)
    
    # Get the period
    r = get_period(phase_int, nbits)
    if r % 2 != 0:
        return None
    
    # Compute factors
    plus = pow(a, r // 2) + 1
    factor1 = gcd(plus, N)

    # Check if factors are trivial
    if factor1 == 1 or factor1 == N:
        return None

    return sorted([factor1, int(N/factor1)])


def factor_number(N: int, a: Optional[int] = None, max_attempts: int = 20) -> Optional[List[int]]:
    """
    Factor an 8-bit integer using Shor's algorithm.

    Args:
        N (int): Number to factor (must be < 256).
        a (Optional[int]): Cofactor.

    Returns:
        Optional[int]: A non-trivial factor of N, or None if failed.
    """

    if a is None:
        a = get_random_base(N)
    
    # run till we get a result or max attempts
    attempts = 0

    while  attempts < max_attempts:
        attempts += 1

        result = run_qpe(a, N)

        if result is not None:
            return result
        
        # get new random base
        a = get_random_base(N)

    warnings.warn(f"Failed to find factors after {max_attempts} attempts.", UserWarning)
    return None





