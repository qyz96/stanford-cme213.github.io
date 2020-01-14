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