{ ... }:

{
  overlays = [(
    finalPkgs: prevPkgs: {
      trilinos-mpi = prevPkgs.trilinos-mpi.overrideAttrs (final: prev: rec {
        version = "14.4.0";

        preConfigure = prev.preConfigure + ''
          cmakeFlagsArray+=(-DTrilinos_ENABLE_ML=ON);
        '';

        src = prev.src.override {
          sha256 = "sha256-jbXQYEyf/p9F2I/I7jP+0/6OOcH5ArFlUk6LHn453qY=";
        };
      });
    }
  )];

  config = {
    allowUnfree = true;
  };
}
