{ pkgs ? import <nixpkgs> {} }:
with pkgs;
mkShell {

  buildInputs = [
    python3
    poetry
    graphviz
  ];

  LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib";

  shellHook = "fish";

}

# let
# myAppEnv = poetry2nix.mkPoetryEnv {
#   projectDir = ./.;
#   editablePackageSources = {
#     my-app = ./src;
#   };
#   overrides = poetry2nix.overrides.withDefaults(self: super: {
#     pyarrow = super.pyarrow.override { preferWheel = true; };
#   });
# };
# in myAppEnv.env
