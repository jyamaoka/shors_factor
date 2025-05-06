import math
import array
import fractions
import numpy as np

def is_prime(n: int) -> bool:
    """
    Check if a number is a prime.

    Args:
        n (int): Number to check.

    Returns:
        bool: True if n is prime, False otherwise.
    """
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def check_if_power(N: int) -> bool:
    """
    Check if N is a perfect power in O(n^3) time, n=ceil(logN) 
    
    Args:
        N (int): Number to check.

    Returns:
        bool: True if n is prime, False otherwise.  
    """
    b=2
    while (2**b) <= N:
        a = 1
        c = N
        while (c-a) >= 2:
            m = int( (a+c)/2 )

            if (m**b) < (N+1):
                p = int( (m**b) )
            else:
                p = int(N+1)

            if int(p) == int(N):
                print('N is {0}^{1}'.format(int(m),int(b)) )
                return True

            if p<N:
                a = int(m)
            else:
                c = int(m)
        b=b+1

    return False

