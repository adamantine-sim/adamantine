attrs @ { callPackage, fetchFromGitHub, ... }:

callPackage ./common.nix (
  rec {
    version = "9.5.2";
    src = fetchFromGitHub {
      owner = "dealii";
      repo  = "dealii";
      rev   = "v${version}";
      hash  = "sha256-m2+1HCAkfY6w3QBT4fuz5dm7E3qurvukRf9nI6xyfpY=";
    };
  } // attrs
)
