---
layout: page
title: Run
nav_order: 3
has_children: true
---

# Run
After compiling *adamantine*, you can run a simulation using
```bash
mpirun -n 2 ./adamantine --input-file=input.info
```
Note that the name of the input file is totally arbitrary, `my_input_file` is as
valid as `input.info`.

There is a [known bug](https://github.com/adamantine-sim/adamantine/issues/130)
when using multithreading. To deactivate multithreading use
```bash
export DEAL_II_NUM_THREADS=1
```
If you use our Docker image, the variable is already set.
