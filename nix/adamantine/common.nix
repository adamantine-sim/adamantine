{
    src, version,

    lib, stdenv, fetchFromGitHub,

    cmake,

    arborx, adiak, caliper, p4est, trilinos-mpi, boost, openmpi, dealii,

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
    "-DCMAKE_CXX_FLAGS=-ffast-math"
    "-DBUILD_SHARED_LIBS=ON"
  ];

  # Manual install if using versions 1.0 since adamantine was lacking CMake installs.
  installPhase = lib.optional (version == "1.0") ''
    mkdir -p $out/bin
    cp bin/adamantine $out/bin
  '';
}

