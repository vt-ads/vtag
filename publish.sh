# Compile
sudo rm -rf dist build
sudo python -m pip install bleach --upgrade
sudo python setup.py sdist bdist_wheel

# Upload to the real PyPI
python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*
