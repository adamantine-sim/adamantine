{
  stdenv, fetchFromGitHub,

  cmake,

  openmpi, trilinos-mpi, arborx, p4est, boost
}:

stdenv.mkDerivation rec {
  pname = "dealii";
  version = "9.6.2";

  src = fetchFromGitHub {
    owner = "dealii";
    repo  = "dealii";
    rev   = "v${version}";
    hash  = "sha256-sIyGSEmGc2JMKwvFRkJJLROUNdLKVhPgfUx1IfjT3dI=";
  };

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    openmpi
    trilinos-mpi
    arborx
    p4est
    boost
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=DebugRelease"
    "-DCMAKE_CXX_STANDARD=17"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DDEAL_II_WITH_TBB=OFF"
    "-DDEAL_II_WITH_64BIT_INDICES=ON"
    "-DDEAL_II_WITH_COMPLEX_VALUES=OFF"
    "-DDEAL_II_WITH_MPI=ON"
    "-DDEAL_II_WITH_P4EST=ON"
    "-DDEAL_II_WITH_ARBORX=ON"
    "-DDEAL_II_WITH_TRILINOS=ON"
    "-DDEAL_II_TRILINOS_WITH_SEACAS=OFF"
    "-DDEAL_II_COMPONENT_EXAMPLES=OFF"
    "-DDEAL_II_WITH_ADOLC=OFF"
    "-DDEAL_II_ALLOW_BUNDLED=OFF"
  ];
}
