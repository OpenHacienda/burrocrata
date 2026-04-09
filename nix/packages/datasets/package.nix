{
  lib,
  python3Packages,
}:
python3Packages.buildPythonApplication {
  pname = "burrocrata-datasets";
  version = "0.1.0";
  pyproject = true;

  src = lib.cleanSource ../../../packages/datasets;

  build-system = [ python3Packages.setuptools ];

  dependencies = with python3Packages; [
    datasets
    python-frontmatter
    click
  ];

  doCheck = false;

  meta = {
    description = "Dataset builders for burrocrata corpora";
    mainProgram = "burrocrata-datasets";
  };
}
