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
