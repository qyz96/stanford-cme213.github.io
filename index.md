---
layout: default
title: "CME 213: Introduction to Parallel Computing using MPI, openMP, and CUDA"
---

Welcome to CME 213!

This class will give hands on experience with programming multicore processors, graphics processing units (GPU), and parallel computers. Focus will be on the message passing interface (MPI, parallel clusters) and the compute unified device architecture (CUDA, GPU).  

Topics will include: network topologies, modeling communication times, collective communication operations, parallel efficiency, MPI, dense linear algebra using MPI. Symmetric multiprocessing (SMP), Pthreads, openMP. CUDA, combining MPI and CUDA, dense linear algebra using CUDA, sort, reduce and scan using CUDA.

Pre-requisites include: C programming language and numerical algorithms (solution of differential equations, linear algebra, Fourier transforms).

### Basic Info

* Location: [School of Education, Room 128](https://campus-map.stanford.edu/?srch=School+of+Education+128)  
* Lectures: Mon/Wed/Fri 1:30-2:50PM
* Instructor: [Eric Darve](https://me.stanford.edu/people/eric-darve)  
* Course Assistants: William Jen, Kingway Liang
* [Syllabus](CME 213 Syllabus.pdf)

### Office Hours

* TBA

### Homework

### Google Cloud Platform 
* [Google Cloud Platform setup instructions](./gcp.html)
* [VM instances information page](https://console.cloud.google.com/compute)
* [Billing page](https://console.cloud.google.com/billing)
* [GCP dashboard](https://console.cloud.google.com/home)

### Lecture notes


### Reading and links

* [OpenACC](https://www.openacc.org/)
* [Compilers that support OpenACC](https://www.openacc.org/tools)
* [OpenACC Specification (ver. 2.7 November 2018)](https://www.openacc.org/sites/default/files/inline-files/OpenACC.2.6.final.pdf)
* [OpenACC Programming and Best Practices Guide](http://www.openacc.org/sites/default/files/OpenACC_Programming_Guide_0.pdf)
* [OpenACC 2.7 API Reference Card](https://www.pgroup.com/lit/literature/openacc-api-guide-2.7.pdf)
* [Thrust programming guide](https://docs.nvidia.com/cuda/thrust/index.html)
* [Thrust examples](https://github.com/thrust/thrust/tree/master/examples)
* [Thrust download page](https://developer.nvidia.com/thrust)
* [Thrust github page](https://thrust.github.io/)
* [Thrust get started guide](https://github.com/thrust/thrust/wiki/Quick-Start-Guide)
* [Presentations and slides on Thrust](https://github.com/thrust/thrust/downloads)
* [CUDA occupancy calculator](Reading/CUDA_Occupancy_Calculator.xls)
* [Kepler GK110/210 whitepaper](Reading/NVIDIA_Kepler_GK110_GK210_Architecture_Whitepaper.pdf) (the K80 uses the GK210 chip)
* [Data sheet for Tesla GPUs](https://en.wikipedia.org/wiki/Nvidia_Tesla). **Tesla** is NVIDIA's brand name for their products targeting stream processing or general-purpose graphics processing units (GPGPU).
* [K80 device info](k80.md)
* [K80 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/TeslaK80-datasheet.pdf)
* [CUDA Programming Guides and References](http://docs.nvidia.com/cuda/index.html)
* [CUDA C Programming Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Programming_Guide.pdf)
* [CUDA C Best Practices Guide](http://docs.nvidia.com/cuda/pdf/CUDA_C_Best_Practices_Guide.pdf)
* [CUDA compiler](https://docs.nvidia.com/cuda/pdf/CUDA_Compiler_Driver_NVCC.pdf)
* [A novel sorting algorithm for many-core architectures based on adaptive bitonic sort](https://ieeexplore.ieee.org/abstract/document/6267838)
* [Adaptive Bitonic Sorting](https://pdfs.semanticscholar.org/bcdf/c4e40c79547c9daf89dada4e1c23056871cb.pdf)
* [OpenMP API Syntax Reference Guide](https://www.openmp.org/wp-content/uploads/OpenMPRef-5.0-111802-web.pdf)
* [C++ threads](http://www.cplusplus.com/reference/thread/thread/)
* [Simple examples of C++ multithreading](https://www.geeksforgeeks.org/multithreading-in-cpp/)
* [LLNL tutorial on Pthreads](https://computing.llnl.gov/tutorials/pthreads/)
* [C++ reference](https://en.cppreference.com/w/cpp)

### Course Schedule

Schedules are tentative and will be updated throughout the quarter.

[See detailed schedule](./schedule.html)

| Week          | Date                 | Topics                             | Homework/Project                            |
| :-----------: | -------------------- | ---------------------------------- | --------------------------                  |
| 1             | Mon, Jan 6           | Introduction and syllabus          |                                             |
| 1             | Wed, Jan 8           | Parallelism, Pthreads              | HW1 out                                     |
| 2             | Wed, Jan 15          | Synchronization                    |                                             |
| 2             | Fri, Jan 17          | OpenMP 1: For loops                | HW1 due <br> HW2 out                        |
| 3             | Wed, Jan 22          | OpenMP 2: Reduction                |                                             |
| 3             | Fri, Jan 24          | OpenMP 3: Shared memory sorting    |                                             |
| 4             | Wed, Jan 29          | CUDA 1                             | HW2 due                                     |
| 4             | Fri, Jan 31          | CUDA 2                             | HW3 out                                     |
| 5             | Wed, Feb 5           | CUDA 3, matrix transpose           |                                             |
| 5             | Fri, Feb 7           | Lecture on homework 4              | HW4 out <br> HW3 due                        |
| 6             | Wed, Feb 12          | Reduction                          |                                             |
| 6             | Fri, Feb 14          | Thrust                             |                                             |
| 7             | Wed, Feb 19          | Lecture on final project           | HW5 out <br> HW4 due <br> Final project out |
| 7             | Fri, Feb 21          | OpenACC by NVIDIA                  |                                             |
| 8             | Wed, Feb 26          | CUDA optimization by NVIDIA        |                                             |
| 8             | Fri, Feb 28          | NVVP by NVIDIA                     |                                             |
| 9             | Wed, Mar 4           | Point-to-point communication       | HW5 due                                     |
| 9             | Fri, Mar 6           | Groups, communicators and topology |                                             |
| 10            | Wed, Mar 11          | -                                  | Project interim report due                  |
| 10            | Fri, Mar 13          | Parallel efficiency                |                                             |
| 10            | Wed, Mar 18          | -                                  | Final project due                           |