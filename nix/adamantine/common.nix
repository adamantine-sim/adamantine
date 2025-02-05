{
    src, version,

    stdenv, fetchFromGitHub,

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
  ];

  installPhase = ''
    mkdir -p $out/bin
    cp bin/adamantine $out/bin
  '';

  doCheck = true;
  check = ''
    ctest -R integration_2d
  '';
}

