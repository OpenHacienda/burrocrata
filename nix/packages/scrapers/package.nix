{
  lib,
  python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "burrocrata-scrapers";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../../../packages/scrapers;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    requests
    beautifulsoup4
    click
    python-frontmatter
  ];

  doCheck = false;

  meta = {
    description = "Scrapers for Spanish tax/legal sources (DGT, ...)";
    mainProgram = "burrocrata-dgt";
  };
}
