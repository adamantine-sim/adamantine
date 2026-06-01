{
  stdenv, fetchFromGitHub,

  cmake,

  openmpi
}:

stdenv.mkDerivation rec {
  pname = "sundials-mpi";
  version = "7.1.1";

  src = fetchFromGitHub {
    owner = "llnl";
    repo  = "sundials";
    rev   = "v${version}";
    hash  = "sha256-W+qwWve3rD3eCxNUP+yB3bAssKzPaCJuS0Pk9FFbtyw=";
  };

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    openmpi
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DENABLE_MPI=ON"
    "-DBUILD_NVECTOR_PARALLEL=ON"
    "-DBUILD_NVECTOR_MPIMANYVECTOR=ON"
    "-DBUILD_SHARED_LIBS=ON"
  ];
}

