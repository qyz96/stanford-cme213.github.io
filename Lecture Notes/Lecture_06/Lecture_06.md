class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME

![:width 40%](Stanford.jpg)

“Optimism is an occupational hazard of programming; feedback is the treatment.”
(Kent Beck)

---
class: center, middle

# Array reduction

A recent addition in OpenMP

[reduction_array.cpp](https://github.com/stanford-cme213/stanford-cme213.github.io/blob/master/Code/Lecture_05/reduction_array.cpp)

---
class: middle

```
#pragma omp parallel for reduction(+: a [0:n])
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            a[j] += b[i][j];
```

---
class: middle

# Array section syntax

`[ lower-bound : length : stride]`

`[:]`

---
class: middle

# Example

`a[lb : n : s]`

Indices: `lb`, `lb+s`, ..., `lb + (n-1)*s`

Example: `a[0:n]`, `a[0:n/2:2]`

---
class: middle, center

Homework 2 focusses on radix sort

---
class: middle, center

Applies to integers or floats only

Use buckets

Partitions the bits into small groups

Order the bits using the buckets

---
class: middle, center

Quicksort

One of the fastest sorting algorithms

---
class: middle, center

On average, it runs very fast, even faster than mergesort.

It requires no additional memory

[Musical demo](https://www.youtube.com/watch?v=9IqV6ZSjuaI)

[Musical demo](https://www.youtube.com/watch?v=8hEyhs3OV1w)

[Musical demo](https://www.youtube.com/watch?v=q4wzJ_uw4aE)

---
class: middle, center

Some disadvantages

Worst-case running time is $O(n^2)$ when input is already sorted

Not stable

---
class: middle

# Quicksort algorithm

Divide and conquer approach

Divide step:

- Choose a pivot x
- Separate sequence into 2 sub-sequences with all elements smaller than x and greater than x

Conquer step:

- Sort the two subsequences

---
class: middle

```
def quicksort(A,l,u):
    if l < u-1:
        x = A[l]
        s = l
        for i in range(l+1,u):
            if A[i] <= x: # Swap entries smaller than pivot
                s = s+1
                A[s], A[i] = A[i], A[s]
        A[s], A[l] = A[l], A[s]
        quicksort(A,l,s)
        quicksort(A,s+1,u)
```

---
class: middle, center

![:width 70%](2020-01-23-14-47-01.png)

---
class: center

![:width 35%](quicksort_par.jpg)

---
class: center

![:width 60%](quicksort_pivot.png)

---
class: middle

# Mergesort

1. Subdivide the list into n sub-lists (each with one element).

2. Sub-lists are progressively merged to produce larger ordered sub-lists.

[Musical demo](https://www.youtube.com/watch?v=ZRPoEKHXTJg)

---
class: center

![:width 55%](mergesort.png)

---
class: middle, center

# Parallel mergesort

When there are many sub-lists to merge, the parallel implementation is straightforward: assign each sub-list to a thread.

When we get few but large sub-lists, the parallel merge becomes difficult.

---
class: middle, center

Merging large chunks

Subdivide the merge into several smaller merges that can be done concurrently.

---
class: middle, center

![:width 90%](mergesort_parallel_merge.gif)

---
class: middle, center

# Bucket and sample sort

---
class: middle

# Bucket sort

Sequence of integers in the interval $[a,b]$

1. Split $[a,b]$ into $p$ sub-intervals
2. Move each element to the appropriate bucket (prefix sum)
3. Sort each bucket in parallel!

---
class: center, middle

# Variant: radix sort

[Musical demo MSD](https://www.youtube.com/watch?v=Tmq1UkL7xeU)

[Musical demo LSD](https://www.youtube.com/watch?v=LyRWppObda4)

---
class: center, middle

Problem: how should we split the interval in bucket sort? 

This process may lead to intervals that are unevenly filled.

Improved version: splitter sort.

---
class: center, middle

![:width 80%](splitter.png)

---
class: center, middle

# Sorting networks

Building block: compare-and-exchange (COEX)

In sorting networks, the sequence of COEX is **independent** of the data

One of their advantages: very regular data access

---
class: middle, center

.compact[*A novel sorting algorithm for many-core architectures based on adaptive bitonic sort,*
H. Peters, O. Schulz-Hildebrandt, N. Luttenberger]

![:width 80%](2020-01-24-08-27-44.png)

---
class: middle, center

# Bitonic sequence

First half &#8599;, second half &#8600;, or

First half &#8600;, second half &#8599;

![:width 70%](2020-01-24-08-32-47.png)

---
class: middle, center

There is a fast algorithm to partially "sort" a bitonic sequence

Bitonic *compare*

---
class: middle, center

Input: bitonic sequence

# Bitonic compare

First half

$\min(E\_0,E\_{n/2}), \min(E\_1,E\_{n/2+1}), \ldots, \min(E\_{n/2-1},E\_{n-1})$

Second half

$\max(E\_0,E\_{n/2}), \max(E\_1,E\_{n/2+1}), \ldots, \max(E\_{n/2-1},E\_{n-1})$

---
class: middle, center

![:width 50%](2020-01-24-08-43-45.png)

---
class: middle, center

Output

Two bitonic sequences

Left is smaller than right

---
class: middle

Build a bitonic sorting network to sort the entire array

Process:

1. Star from small bitonic sequences
2. Use compare and merge to get longer bitonic sequences
3. Repeat until sorted

---
class: middle, center

![](bitonic_sort.png)

---
class: middle, center

Complexity

$(\log n)^2$ passes

[Musical demo](https://www.youtube.com/watch?v=r-erNO-WICo)

---
class: middle

Exercise

- bitonic_sort_lab.cppOpen this code to start the exercise
- bitonic_sort.cppSolution with OpenMP
- bitonic_sort_seq.cppReference sequential implementation
