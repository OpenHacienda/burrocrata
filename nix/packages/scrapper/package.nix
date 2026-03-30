{
  lib,
  python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "scrapper-dgt";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../../../python;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    requests
    beautifulsoup4
    click
    python-frontmatter
  ];

  doCheck = false;

  meta = {
    description = "Scraper for DGT consultas vinculantes from PETETE";
    mainProgram = "scrapper-dgt";
  };
}
