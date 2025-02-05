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

    packages = with config; let
      callPackage = lib.callPackage libs;

      libs = {
        adiak   = callPackage ./nix/dependencies/adiak.nix   {};
        caliper = callPackage ./nix/dependencies/caliper.nix {};
        arborx  = callPackage ./nix/dependencies/arborx.nix  {};
        dealii  = callPackage ./nix/dependencies/dealii.nix  {};
      };
    in rec {
      default = adamantine.devel;

      inherit libs;

      adamantine = rec {
        devel = callPackage ./nix/adamantine/common.nix {
          version = self.shortRev or self.dirtyShortRev;
          src     = self;
        };

        stable = callPackage ./nix/adamantine/stable.nix { inherit callPackage; };
      };
    };

    devShells = with config; rec {
      default = adamantineDev;

      adamantineDev = pkgs.mkShell rec {
        name = "adamantine-dev";

        packages = with pkgs; [
          git
          gdb
          clang-tools
          ninja
        ] ++ pkgs.lib.optionals (pkgs.stdenv.hostPlatform.isLinux) [
          cntr
        ] ++ self.outputs.packages.${system}.adamantine.devel.buildInputs
          ++ self.outputs.packages.${system}.adamantine.devel.nativeBuildInputs
          ++ self.outputs.packages.${system}.adamantine.devel.propagatedBuildInputs;

        # For dev, we want to disable hardening.
        hardeningDisable = [
          "bindnow"
          "format"
          "fortify"
          "fortify3"
          "pic"
          "relro"
          "stackprotector"
          "strictoverflow"
        ];

        # Ensure the locales point at the correct archive location.
        LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
      };
    };
  });
}
