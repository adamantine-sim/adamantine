## See NIX.md for help getting started with Nix

{
  description = "Software to simulate heat transfer for additive manufacturing";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-24.11";
    utils.url   = "github:numtide/flake-utils";
  };

  outputs = inputs @ { self, utils, ... }: utils.lib.eachDefaultSystem (system: rec {
    config = rec {
      pkgs = import inputs.nixpkgs {
        inherit system;
        inherit (import ./nix/nixpkgs/config.nix {}) overlays config;
      };
    };

    lib = with config; {
      callPackage = set: pkgs.lib.callPackageWith (pkgs // set);
    };

    packages = with config; rec {
      libs = let
        callPackage = lib.callPackage libs;
      in {
        adiak   = callPackage ./nix/dependencies/adiak.nix   {};
        caliper = callPackage ./nix/dependencies/caliper.nix {};
        arborx  = callPackage ./nix/dependencies/arborx.nix  {};
        dealii  = callPackage ./nix/dependencies/dealii.nix  {};
      };

      adamantine = (lib.callPackage libs) ./nix/adamantine/common.nix { version = self.dirtyShortRev; src = self; };
    };

    devShells = with config; rec {
      # TODO
    };
  });
}
