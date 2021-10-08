# Lab exercises for COMP0026 Image Processing

## TL;DR

* Install [Git](https://git-scm.com) (if you don't already have it)
* Install [Miniconda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) (if you don't already have it) 
* Clone this repository to your local machine:
    ```sh
    git clone https://github.com/LouAgapito/ImPro26.git
    ```
* Create and activate a virtual environment:
    ```
    cd labs
    conda create -n myenv
    conda activate myenv
    ```
* Install Python package requirements:
    ```sh
    # after you activate the environment
    pip install -r requirements.txt
    ```
* Start jupyter server
  ```sh
  # move to the correct directory
  cd notebooks_dir/
  # start the server
  jupyter lab
  # or
  jupyter notebook
  ```
* Read the week's lab exercises notebook `week_N.ipynb`
* Add your code to the script `week_N.ipynb`
* Run each cell and make sure you have the correct answer and add your comments as a markdown if needed

## About

This repository contains lab exercises for the [COMP0026 Image Processing](https://moodle.ucl.ac.uk/enrol/index.php?id=1381) module for taught undergrad and MSc students at UCL, delivered in Autumn 2021. 
Exercises are designed to be attempted in the on-campus lab sessions on Thursday mornings, though you are free to do additional work in your own time if you wish.

Lab attendance will be monitored, but the exercises are **not graded**. 
They are intended as learning experiences, to help you understand and apply different image processing algorithms. 
You are welcome to discuss and help each other with these tasks and to ask for assistance and clarification from the TAs, but there is nothing to be gained by simply copying each others' work.

### Contents

Exercises for week *N* are specified in notebook `week_N.ipynb`.
You should add your solution code and comments about the result to this file.
The code cell can be run using _run cell_ button or using jupyter notebook keyboard shortcut.

Note that the shortcuts are for Windows and Linux users.
Anyway, for the Mac users, they’re different buttons for Ctrl, Shift, and Alt:
* `Ctrl:`: command key ⌘
* `Shift:`: Shift ⇧
* `Alt:`: option ⌥

List of useful shortcuts:
* `Shift + Enter`: run the current cell, select below
* `Ctrl + Enter`:  run selected cells
* `Alt + Enter`:   run the current cell, insert below
* `Ctrl + S`:      save and checkpoint


In addition to the spec and script for each week, there are a few other files and folders in the repo:

* `images`: a folder contains several images that can be used as input to the routines you will implement.
* `README.md`: this file.
* `utils.py`: a library of utility functions that are used by the script code, or are available to be used by your own solution code.
* `requirements.txt`: a list of additional Python packages to install.
* `LICENSE`: text of the MIT License that applies to all code and documentation in this repository. (Summary: in the unlikely event that you have any reason to do so, you are free to reuse this material for any purpose you like.)


## Cloning & Updating

In order to copy the repo you need to have a working installation of the Git version control system. If you don't already have this, you can [download it here](https://git-scm.com).

Choose a convenient location for your working directory and download an initial copy of the repository with `git`:
```sh
git clone https://github.com/LouAgapito/ImPro26.git
```
New lab exercises will be added to the repo each week. You can download updates with the following commands:
```sh
git fetch
git merge origin/main
```
Note however that `git` may report issues when you try to merge our upstream changes with your own if you have uncommitted changes in your directory, or if you have made changes that conflict with changes we have made in the main repo.

We will do our best to avoid making any changes that are likely to cause conflicts. You can generally assume that the `week_N.ipynb` files will not be updated and can be freely edited and committed. (You should not assume this about `utils.py` or `requirements.txt` -- try to avoid editing these files if you can.) If conflicting changes do become necessary, for example to fix significant errors in one of the supplied scripts, we will announce it on Moodle and explain what you'll need to do about it.

We recommend using `git` to track your own changes as you work on the exercises. Commit your work at appropriate intervals and only `fetch/merge` new changes when your own changes are up to date.

However, if you have made changes that you don't want to commit for some reason, you can also temporarily get them out of the way using [`git stash`](https://git-scm.com/book/en/v2/Git-Tools-Stashing-and-Cleaning):
```sh
git stash push
git fetch
git merge origin/main
git stash apply
```

## Python Setup

The exercises require a local installation of Python 3, along with a number of additional packages for numerical programming, plotting and image processing. We suggest using [Anaconda](https://www.anaconda.com/products/individual-d) / [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with the latest stable release of Python (currently 3.9.7) as it already have most of the required packages already installed in its' default environment. If you know and prefer [python.org](https://www.python.org/downloads/) you are welcome to use that instead.

(Although we recommend the latest Python, the code has also been tested on Python 3.6.8, which is the version currently installed on some of the CS lab machines. It is possible, albeit suboptimal, to set up and run the exercises on one of those machines via SSH.)

### Virtual Environments

The package requirements for the lab exercises are pretty vanilla, but we strongly recommend working in a dedicated [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) in order to avoid any conflicts or compatibility issues with any other Python work you may be doing.

There are several options for how and where to set up such a virtual environment.
If you already have experience doing so then feel free to use any configuration you are comfortable with. If you haven't done this before and/or would rather not think about it, follow the default setup instructions below.

#### Default virtual environment setup

A straightforward way to configure your virtual environment is to store it in a hidden subdirectory of your working directory (ie, the directory containing this repository). Once you have cloned the repo as described above, change into the directory:
```sh
cd labs
```
Initialise a new virtual environment:
```sh
conda create -n myenv python=3.6
```
This will creat environment named myenv with Python 3.6 (you can specify python version as you like). 
But the environment will be saved in `/envs/` and no packages will be installed.
To view current environment you can use `conda env list` and to activate the environment you can use:
```sh
conda activate myenv
```

When the virtual environment is active, your commmand prompt will be modified with the prefix `(myenv)`.

### Installing Required Packages

With your virtual environment active, you should be able to install all required packages using `pip`, like this:
```sh
pip install -r requirements.txt 
```
or you can use conda to install the packages like this:
```sh
conda install [packeg_name] 
```

## Working Environment

If you are an experienced Python coder with a preferred development environment, you should use that. 
If you don't have any existing preference, we recommend using [jupyter notebooks](https://jupyter.org/) it is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. 
Also, you can use a Python-aware editor such as the [PyCharm IDE](https://www.jetbrains.com/pycharm/) (the Pro version is free to students), together with an [IPython](https://ipython.readthedocs.io/) interactive session running in a terminal window for testing and debugging or [VS Code](https://code.visualstudio.com) (be sure to install the [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)).


## Feedback

Please post questions, comments, issues or bug reports to the [COMP0026 Moodle forum]() or raise them with the TAs during your lab sessions.

