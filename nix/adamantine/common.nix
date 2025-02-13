{
    src, version,

    lib, stdenv,

    cmake,

    arborx, adiak, caliper, p4est, trilinos-mpi, boost, openmpi, dealii,

    doCheck ? true,

    # Allow extra args as needed for callPackage chaining - not ideal.
    ...
}:

stdenv.mkDerivation rec {
  pname = "adamantine";
  inherit version;

  inherit src;

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    arborx
    adiak
    caliper
    p4est
    trilinos-mpi
    boost
    dealii
  ];

  propagatedBuildInputs = [
    openmpi
  ];

  cmakeFlags = [
    "-DADAMANTINE_ENABLE_ADIAK=ON"
    "-DADAMANTINE_ENABLE_CALIPER=ON"
  ] ++ lib.optionals (doCheck) [
    "-DADAMANTINE_ENABLE_TESTS=ON"
  ];

  # Manual install if using versions 1.0 since adamantine was lacking CMake installs.
  installPhase = lib.optional (version == "1.0") ''
    mkdir -p $out/bin
    cp bin/adamantine $out/bin
  '';

  inherit doCheck;
  checkPhase = ''
    ctest -R integration_2d
  '';
}

