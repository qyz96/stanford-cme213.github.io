class: center, middle

# CME 213, ME 339 - Winter 2020

## Eric Darve, ICME, Stanford

![:width 40%](Stanford.jpg)

“Controlling complexity is the essence of computer programming.”
(Brian Kernigan)

---
class: center, middle

# Google Cloud Platform

Instructions for setup

https://stanford-cme213.github.io/gcp.html

---
class: middle

# Google Cloud Platform

To do:

1. Use a **non** a Stanford Google account, e.g., personal
2. Redeem coupons sent by TAs

---
class: middle

# Setup

1. Set-up GCP project</br>https://console.cloud.google.com/cloud-resource-manager
2. Install GCP SDK on your local machine</br>https://cloud.google.com/sdk/docs/downloads-interactive

---
class: middle

# Create virtual machine `VM`

1. Download script</br>.compact[https://stanford-cme213.github.io/Code/create_vm_openmp.sh]
2. Run `./create_vm_openmp.sh`

---
class: center, middle

# The omp VM

Runs Linux Ubuntu

8 cores

`make` & `g++`

---
class: middle

# Log in and copy files

Log in

`$ gcloud compute ssh omp`

Copy

`$ gcloud compute scp LOCAL_PATH omp:VM_PATH`

---
class: center, middle

# OpenMP

---
class: center, middle

C++ threads are great for low-level multicore programming

But too general and complicated for engineering applications

---
class: middle

Two common scenarios

1. `For` loop: partition the loop into chunks</br>Have each thread process one chunk.
2. Hand-off a block of code to a separate thread

---
class: center, middle

OpenMP simplifies the programming significantly.

In many cases, adding one line is sufficient to make it run in parallel.

---
class: center, middle

OpenMP is the standard approach in scientific computing for multicore processors

---
class: center, middle

# What is OpenMP?

Application Programming Interface (API)

Jointly defined by a group of major computer hardware and software vendors

---
class: center, middle

Portable, scalable model for developers of shared memory parallel applications

Supports C/C++ and Fortran on a wide variety of computers

---
class: center, middle

OpenMP website</br>https://openmp.org 

Wikipedia</br>https://en.wikipedia.org/wiki/OpenMP

LLNL tutorial</br>https://computing.llnl.gov/tutorials/openMP/

---
class: center, middle

![:width 50%](2020-01-15-10-49-32.png)

---
class: center, middle

Reference

https://software.intel.com/sites/default/files/managed/6a/78/parallel_mag_issue18.pdf

---
class: center, middle

# Compiling your code

Header file:

```
#include <omp.h>
```

---
class: center, middle

Compiler | Flag
--- | ---
gcc</br>g++</br>g77</br>gfortran | `-fopenmp`
icc</br>icpc</br>ifort | `-openmp`

---
class: center, middle

# Parallel regions

---
class: center, middle

# `#pragma omp parallel`

This is the basic building block

This is often not how OpenMP is used in practice

Will serve simply as an introduction

---
class: center, middle

![:width 100%](2020-01-15-11-04-57.png)

---
class: center, middle

`#pragma omp parallel`

The block of code that follows is executed by all threads in the team

---
class: center, middle

# Computing $\pi$

![:width 30%](pi.png)

---
class: center, middle

# Formula used to approximate $\pi$

$$ \frac{\pi}{2} = 1 + \frac{1}{3} \Big(
    1 + \frac{2}{5} \Big( 
        1 + \frac{3}{7} \Big(
            1 + \frac{4}{9} \Big(
            1 + \cdots
            \Big)
        \Big)
    \Big)
\Big)$$

---
class: middle

```
#pragma omp parallel num_threads(nthreads)
{
    long tid = omp_get_thread_num();
    // Only thread 0 does this
    if (tid == 0)
    {
        int n_threads = omp_get_num_threads();
        printf("[info] Number of threads = %d\n", n_threads);
    }
    // Print the thread ID
    printf("Hello World from thread = %ld\n", tid);

    // Compute digits of pi
    DoWork(tid, ndigits[tid], etime[tid]);
}
// All threads join the master thread and terminate
```

---
class: middle

Choose your compiler in `Makefile`

`$ make`

`$ ./hello_world_openmp`

---
class: center, middle

# Common use case: `for` loop

This example cover 99% of the needs for scientific computing