attrs @ { callPackage, fetchFromGitHub, ... }:

callPackage ./common.nix (
  rec {
    version = "9.6.2";
    src = fetchFromGitHub {
      owner = "dealii";
      repo  = "dealii";
      rev   = "v${version}";
      hash  = "sha256-sIyGSEmGc2JMKwvFRkJJLROUNdLKVhPgfUx1IfjT3dI=";
    };
  } // attrs
)
