{
  stdenv, fetchFromGitHub,

  cmake,

  openmpi, trilinos-mpi
}:

stdenv.mkDerivation rec {
  pname = "arborx";
  version = "1.5";

  src = fetchFromGitHub {
    owner = "arborx";
    repo  = "ArborX";
    rev   = "v${version}";
    hash  = "sha256-XhvWKex7sKACY90emvV9uGw/ACI00dLq1HoeG2nWthk=";
  };

  nativeBuildInputs = [
    cmake
  ];

  buildInputs = [
    openmpi
    trilinos-mpi
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_CXX_EXTENSIONS=OFF"
    "-DARBORX_ENABLE_MPI=ON"
  ];
}

