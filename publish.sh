# Compile
rm -rf dist build
python -m pip install bleach --upgrade
python setup.py sdist bdist_wheel

# Upload to the real PyPI
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
