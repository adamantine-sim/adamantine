{
    stdenv, fetchFromGitHub,

    cmake, python3,

    adiak, openmpi
}:

stdenv.mkDerivation rec {
  pname = "caliper";
  version = "2.10.0";

  src = fetchFromGitHub {
    owner = "LLNL";
    repo  = "caliper";
    rev   = "v${version}";
    hash  = "sha256-4rnPbRYtciohLnThtonTrUBO+d8vyWvJsCgoqbJI5Rg=";

    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
    python3
  ];

  buildInputs = [
    adiak
    openmpi
  ];

  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SHARED_LIBS=ON"
    "-DWITH_ADIAK=ON"
    "-DWITH_LIBDW=ON"
    "-DWITH_LIBPFM=ON"
    "-DWITH_LIBUNWIND=ON"
    "-DWITH_MPI=ON"
    "-DWITH_SAMPLER=ON"
  ];
}
