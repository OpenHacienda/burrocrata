_: {
  # The root file that marks the top of the project tree.
  projectRootFile = "flake.nix";

  # Nix
  programs.nixfmt.enable = true;

  # Python — ruff handles both formatting and import sorting.
  programs.ruff-format.enable = true;
  programs.ruff-check.enable = true;
}
