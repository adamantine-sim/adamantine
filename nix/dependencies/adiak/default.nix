{
  stdenv, fetchFromGitHub,

  cmake
}:

stdenv.mkDerivation rec {
  pname = "adiak";
  version = "0.5.0";

  src = fetchFromGitHub {
    owner = "LLNL";
    repo  = "Adiak";
    rev   = "v${version}";
    hash  = "sha256-8cBQINKOU2gRIljaLy9TPqZP3UiYl+8GmTIoRJZNDA8=";

    fetchSubmodules = true;
  };

  nativeBuildInputs = [
    cmake
  ];

  cmakeFlags = [
    "-DBUILD_SHARED_LIBS=ON"
    "-DCMAKE_BUILD_TYPE=Release"
  ];
}
