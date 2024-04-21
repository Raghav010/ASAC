#!/bin/bash
rm .gitignore
find * -size +50M | sed 's|^\./||g' | cat >> .gitignore
