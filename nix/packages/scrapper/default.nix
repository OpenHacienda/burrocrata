{
  pkgs,
  ...
}:
pkgs.python3Packages.buildPythonApplication {
  pname = "scrapper-dgt";
  version = "0.1.0";
  pyproject = true;

  src = ../../../python;

  build-system = [ pkgs.python3Packages.setuptools ];

  dependencies = with pkgs.python3Packages; [
    requests
    beautifulsoup4
    click
    pyyaml
  ];

  # The test suite requires network access (hits the real PETETE server),
  # so we skip it inside the Nix sandbox.
  doCheck = false;

  meta = {
    description = "Scraper for DGT consultas vinculantes from PETETE";
    mainProgram = "scrapper-dgt";
  };
}
