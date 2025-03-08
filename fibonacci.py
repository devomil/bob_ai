def calculate_fibonacci(n):
    """Calculate the Fibonacci sequence up to the nth number."""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]

    sequence = [0, 1]
    for _ in range(2, n):
        sequence.append(sequence[-1] + sequence[-2])
    return sequence

if __name__ == "__main__":
    print("Fibonacci sequence for n=10:", calculate_fibonacci(10))
