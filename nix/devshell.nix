{ pkgs, inputs, ... }:
let
  treefmtEval = inputs.treefmt.lib.evalModule pkgs ./treefmt.nix;
in
pkgs.mkShell {
  packages = [
    # Python runtime — add per-project packages inside each sub-package instead.
    pkgs.python3

    # Formatter / linter (same binary used by treefmt)
    pkgs.ruff

    # Dependency / virtualenv management
    pkgs.uv

    # Code-quality helpers
    pkgs.pyright # static type checker

    # Nix tooling
    pkgs.nixfmt-rfc-style
    pkgs.nil # Nix language server

    # treefmt wrapper so developers can run `treefmt` directly
    treefmtEval.config.build.wrapper
  ];

  # Keep the shell tidy: don't leak Python bytecode into the repo.
  env = {
    PYTHONDONTWRITEBYTECODE = "1";
  };

  shellHook = ''
    echo "🐍  Python monorepo dev-shell ready"
    echo "   python  → $(python3 --version)"
    echo "   ruff    → $(ruff --version)"
    echo "   uv      → $(uv --version)"
    echo "   pyright → $(pyright --version)"
    echo ""
    echo "   run \`treefmt\` to format everything"
  '';
}
