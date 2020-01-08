class: center, middle

# CME 213, ME 339 - Winter 2020</br>Introduction to parallel computing using MPI, openMP, and CUDA

## Eric Darve, ICME, Stanford

![:scale 40%](Stanford.jpg)

"The city's central computer told you? R2D2, you know better than to trust a strange computer!" (C3PO)

---
class: center, middle

# Instructors

Eric Darve, ME, ICME, darve@stanford.edu

NVIDIA engineers guest lectures

![:scale 20%](NVLogo_2D.jpg)

---
class: center, middle

# Teaching assistants

William Jen (head TA)

![:scale 30%](WJ.jpg)

---
class: center, middle

# Teaching assistants

Kingway Liang

![:scale 30%](KL.jpg)

---
class: center, middle

Classes will be primarily on **Wednesday and Friday**

---
class: center, middle

**C++ refresher** with William

This Friday 1:30PM in School of Education 128

---
class: center, middle

# Class web page

https://stanford-cme213.github.io/

Class material, homework, final project

---
class: center, middle

# Office Hours

See https://stanford-cme213.github.io/

W & Th 7-9PM with TA

Th 9AM-10:45AM with instructor</br>
Bldg 520, room 120

---
class: center, middle

# Canvas

https://canvas.stanford.edu/

Discussion forum site, announcements, syllabus

---
class: center, middle

# Homework grading and regrades

Gradescope

https://gradescope.com

Entry code 9P3654

---
class: center, middle

# How to get support? 

Online discussion forum on canvas

https://canvas.stanford.edu/courses/110850/discussion_topics

---
class: center, middle

Please be civil on the forum

No disrespectful or demeaning posts

![:scale 30%](2020-01-05-16-03-48.png)

---
class: middle

# Grading, homework, project

- 1 pre-requisites homework + 4 homework assignments: 65% of grade
- One final project: 35% of grade

---
class: center, middle

# Final Project

Deep neural network to recognize hand-written digits

Will involve CUDA and MPI programming

![:scale 60%](MnistExamples.png)

---
class: center, middle

![:scale 100%](2020-01-06-12-55-25.png)

---
class: center, middle

DNN playground

https://playground.tensorflow.org/

---
class: center, middle

![:scale 70%](alphago.png)

AlphaGo played a handful of highly inventive winning moves, several of which were so surprising they overturned hundreds
of years of received wisdom.

---
class: center, middle

# DNN relies on parallel computing

Program | Hardware
--- | ---
AlphaGo Fan | 176 GPUs
AlphaGo Lee | 48 TPUs
AlphaGo Master | Single machine with 4 TPUs
AlphaGo Zero | Single machine with 4 TPUs

---
class: center, middle

# Computer access

Google Cloud Platform Education Grant

Coupons will be handed out

https://console.cloud.google.com/

![:scale 30%](https://cloud.google.com/_static/e0c37e1e84/images/cloud/cloud-logo.svg)

---
class: center, middle

# Books!

Available electronically 

http://searchworks.stanford.edu/

![:scale 100%](2020-01-05-18-15-58.png)

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/412KPqS78qL._SX313_BO1,204,203,200_.jpg)]

.two-column[# General parallel computing books
Parallel Programming for Multicore and Cluster Systems, by Rauber and R&uuml;nger

Applications focus mostly on linear algebra]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51dMzTYk0uL._SX307_BO1,204,203,200_.jpg)]
.two-column[Introduction to Parallel Computing, by Grama, Gupta, Karypis, Kumar

Wide range of applications from sort to FFT, linear algebra and tree search
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51Ms5SrUDbL._SX403_BO1,204,203,200_.jpg)]
.two-column[An introduction to parallel programming, Pacheco. More examples and less theoretical

Applications include N-body codes and tree search
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51zw7gFGGsL._SX442_BO1,204,203,200_.jpg)]
.two-column[# OpenMP and multicore books

Using OpenMP: portable shared memory parallel programming, by Chapman, Jost, van der Pas

Advanced coverage of OpenMP
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/511A3lKmX4L._SX403_BO1,204,203,200_.jpg)]
.two-column[The art of multiprocessor programming, by Herlihy, Shavit

Specializes on advanced multicore programming
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51O3HRtQVVL._SX438_BO1,204,203,200_.jpg)]
.two-column[Using OpenMP-The Next Step: Affinity, Accelerators, Tasking, and SIMD, by van der Pas, Stotzer, Terbo

Covers recent extensions to OpenMP and some advanced usage
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/514h7rF2XjL._SX396_BO1,204,203,200_.jpg)]
.two-column[# CUDA books

Professional CUDA C Programming, by Cheng, Grossman, McKercher

**Recommended for this class;** has more advanced usage like multi-GPU programming
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51eBOpA9WiL._SX404_BO1,204,203,200_.jpg)]
.two-column[Programming Massively Parallel Processors: A Hands-on Approach, by Kirk, Hwu

In its 3rd edition now; covers a wide range of topics including numerical linear algebra, applications, parallel programming patterns
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/513f5%2BZ10sL._SX407_BO1,204,203,200_.jpg)]
.two-column[CUDA Handbook: A Comprehensive Guide to GPU Programming, by Wilt

Lots of advanced technical details on memory, streaming, the CUDA compiler, examples of CUDA optimizations
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/41noCqxvPDL._SX403_BO1,204,203,200_.jpg)]
.two-column[CUDA Programming: A Developer's Guide to Parallel Computing with GPUs, by Cook

Extensive CUDA optimization guide; practical tips for debugging, memory leaks
]

---
class: middle

# CUDA online documentation **(preferred)**

Programming guides and API references</br>http://docs.nvidia.com/cuda/index.html

Teaching and learning resources from NVIDIA</br>https://developer.nvidia.com/cuda-education-training

Reading material on class page:
- CUDA_C_Best_Practices_Guide.pdf
- CUDA_C_Programming_Guide.pdf



---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/51lqHdfG8aL._SX397_BO1,204,203,200_.jpg)]
.two-column[# MPI books

Parallel Programming with MPI, by Pacheco

Classic reference; somewhat dated at this point
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/41WVTvWr7TL._SX442_BO1,204,203,200_.jpg)]
.two-column[Using MPI: Portable Parallel Programming with the Message-Passing Interface, by Gropp, Lusk, Skjellum

Very complete reference
]

---

.two-column[![:scale 70%](https://images-na.ssl-images-amazon.com/images/I/41oPYfXMvBL._SX442_BO1,204,203,200_.jpg)]
.two-column[Using Advanced MPI: Modern Features of the Message-Passing Interface, by Gropp, Hoefler, Thakur, Lusk

Same authors as previous entry; discusses recent and more advanced features of MPI
]

---
class: center, middle

# What this class is about

Multicore processors; Pthreads, C++ threads, OpenMP

NVIDIA graphics processors using CUDA

Computer clusters using MPI


---
class: center, middle

Numerical algorithms for illustration

Sort, linear algebra, basic parallel primitives

![:scale 30%](2020-01-05-18-17-53.png)

---
class: center, middle

# What this class is **not** about

Parallel computer architecture

Parallel design patterns and programming models

Parallel numerical algorithms

![:scale 30%](2020-01-05-18-19-22.png)

---
class: center, middle

# Other related classes

*CME 342: Parallel Methods in Numerical Analysis;* parallel algorithms

*CS 149: Parallel Computing;* </br>hardware, synchronization mechanisms, parallel programming models

---
class: center, middle

*EE 382A: Parallel Processors Beyond Multicore Processing* 

SIMD programming, parallel sorting with sorting networks,
string comparison with dynamic programming, arbitrary-precision operations with fixed-point numbers

---
class: center, middle

# Requirements and pre-requisites

**Basic knowledge of UNIX**

ssh, compilers, makefile, git

---
class: center, middle

**Knowledge of C and C++**

Pointers, memory, templates, polymorphism, standard library

---
class: center, middle

**General proficiency in scientific programming**

Testing, verification, and debugging

---
class: center, middle

# Students with Documented Disabilities

Office of Accessible Education (OAE)

563 Salvatierra Walk; 723-1066

http://oae.stanford.edu

**Let us know right away!**

![:scale 30%](2020-01-05-18-20-03.png)

---
class: middle

# Honor Code and Office of Community Standards

Violations include at least the following circumstances: copying material from 
- another student,
- previous year solution sets,
- solutions found on the internet

.center[![:scale 20%](2020-01-05-18-06-51.png)]

---
class: middle, center

Do not post any material from this class online

.center[![:scale 20%](2020-01-05-18-08-04.png)]

---
class: middle, center

If found guilty of a violation, your grade will be automatically lowered by at least one letter grade, and the
instructor may decide to give you a </br>"No Credit" grade.

.center[![:scale 30%](2020-01-05-18-09-22.png)]