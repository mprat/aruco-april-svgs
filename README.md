# aruco-april-svgs
Generate Aruco/April tags in SVG format from OpenCV

# Installation for Local Development

```
pyenv virtualenv 3.12 aruco-april-svgs
pyenv activate aruco-april-svgs
pip install .
```

To generate all the tags locally:

```
pyenv activate aruco-april-svgs
python src/tags/generate.py
```

This will generate all the relevant tags into an `output/` folder in the root of the directory.
