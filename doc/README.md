This is the documentation for `adamantine` **developers**. The **user documentation** can
be found at https://adamantine-sim.github.io/adamantine

# Speed up compilation during development
Due to the large number of template parameters, compiling `adamantine` can be
time-consuming. You can significantly speed up compilation during
development by temporarily restricting the instantiated parameters.

## Steps to optimize compilation
1. **Modify `source/instantiation.hh`**:
   * Comment out `ADAMANTINE_P_ORDER` for orders 1 and above.
   * Comment out `ADAMANTINE_FE_DEGREE` for orders 2 and above.
   * Comment out `(dealii::QGaussLobatto<1>)` within `ADAMANTINE_QUADRATURE_TYPE`.

   > **Note:** These changes limit the number of specific templated instantiations the compiler needs to generate, reducing the overall compilation time.

2. **Disable Tests**:
   Since the `source/instantiation.hh` file has been modified to restrict instantiations, you will be disable the tests as they rely on the full set of instantiations. Ensure tests are disabled in your build process when using this temporary configuration.

3. **Compile only the library**:
   `adamantine` is a driver built on top an internal library `Adamantine`. The reduced instantiations prevent you to compile `adamantine` but you can still compile the library using:
   ```bash
   make -jN Adamantine
   ```
