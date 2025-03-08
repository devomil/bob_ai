def bubble_sort(arr):
    """Sort a list using Bubble Sort."""
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    """Sort a list using Quick Sort."""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

if __name__ == "__main__":
    print("Bubble Sort:", bubble_sort([64, 34, 25, 12, 22, 11, 90]))
    print("Quick Sort:", quick_sort([64, 34, 25, 12, 22, 11, 90]))
