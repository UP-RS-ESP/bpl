# bpl
Python module to work with bounded power-law (BPL) distributed random variables.

This module contains a few basic functions that help analyse power-law
distributed random variables that can be either bounded only from below or
from both above and below. It provides the following functions:

1.  Generate random samples from power-law distributed random variables.
2.  Estimate a logarithimically binned histogram for a power-law
    distributed random variable.
3.  Estimate the probability density function and the cumulative
    distribution function for specified power-law exponent.

# Installation

1. Simply clone this repository and copy ``bpl.py`` in the directory where you are
working in.

2. Alternatively, you can clone this repository to a desired directory of your
choice and then create and Python startup file ``.pythonstartup`` in your 
home directory as:
```python
import sys
import os
home = os.path.expanduser("~")
sys.path.append(home + "/<path_to_package>")
```
Save the file, and that's it! From then on, all python shells should be able to
detect (and import) the ``bpl`` package without any problem.

# Documentation

An overview of the functions and its usage can be found at:
http://up-rs-esp.github.io/bpl

# Examples

An example script ``example.py`` is included in the repository. 

# License

Copyright (c) 2016 Bedartha Goswami

This program is free software; you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301,USA.

# Author

Bedartha Goswami <goswami@uni-potsdam.de>

# About this file

Created: Tue May 24, 2016  09:42PM

Last modified: Wed May 25, 2016  10:16PM



