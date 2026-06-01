# Minimal Makefile for Sphinx documentation.

SPHINXBUILD ?= python -m sphinx
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help html clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html"

clean:
	rm -rf "$(BUILDDIR)"
