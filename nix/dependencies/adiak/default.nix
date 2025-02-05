{
  stdenv, fetchFromGitHub,

  cmake
}:

stdenv.mkDerivation rec {
  pname = "adiak";
  version = "0.4.0";

  src = fetchFromGitHub {
    owner = "LLNL";
    repo  = "Adiak";
    rev   = "v${version}";
    hash  = "sha256-S4ZLU6f/njdZXyoQdCJIDzpQTSmfapZiRe4zIex5f0Q=";

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

