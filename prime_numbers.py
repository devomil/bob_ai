def generate_primes(n):
    """Generate a list of prime numbers up to n."""
    primes = []
    for num in range(2, n + 1):
        is_prime = all(num % i != 0 for i in range(2, int(num**0.5) + 1))
        if is_prime:
            primes.append(num)
    return primes

if __name__ == "__main__":
    print("Prime numbers up to 50:", generate_primes(50))
