# Adapted from https://github.com/huggingface/kernels-community/blob/18153cd077b6541e9c4cb98ad9678db4e1253d1f/flake.nix
# Changes: renames the flake description and dev shell to kernels-community-rs.

{
  description = "kernels-community-rs tooling";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.follows = "kernels/nixpkgs";
    kernels.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      flake-utils,
      kernels,
      nixpkgs,
    }:
    let
      systems = with flake-utils.lib.system; [
        aarch64-darwin
        aarch64-linux
        x86_64-linux
      ];
    in
    flake-utils.lib.eachSystem systems (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        devShell =
          with pkgs;
          mkShell {
            name = "kernels-community-rs-dev-shell";
            nativeBuildInputs = [
              pinact
            ];
          };

        formatter = pkgs.nixfmt-tree;
      }
    );
}
