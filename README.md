# Automated Walk Bike Counter

## About

The City currently does bicycle and pedestrian counts via having a person manually count
the number of cyclists and pedestrians that go through an intersection,
either in person or by viewing a video recording.

However, thanks to advances in computer vision we can now automate that,
allowing us to constantly count the number of pedestrians and cyclists,
rather than sampling a (possibly not representative) time and location.

This project is a Python application that implements a
[computer vision algorithm](https://pdfs.semanticscholar.org/c1d9/8fca75c63fd5975fc2fcd3fe07ac02de4a5b.pdf)
developed by CSULA that allows you to run the pipeline on your own cameras feeds.

This approach allows cities and others to implement CV pedestrian and bicycle counting techniques on existing camera infrastructure.

## Sponsors

This work has been generously sponsored by a
[grant](https://ladot.lacity.org/sites/g/files/wph266/f/Press%20Release%20LADOT%20Awarded%20Mobility%20Grant%2C%20Will%20Conduct%20Department%27s%20First%20Count%20of%20Walkers%20and%20Bicyclists.pdf) from the Toyota Mobility Foundation (TMF).

## Partners

#### CSULA

Dr. Mohammad Pourhomayoun, Moahmmad Vahedi, Haiyan Wang.

#### Los Angeles DOT/ITA

Hunter Owens, Ian Rose, Janna Smith, Anthony Lyons.

## Goals

Allow us to know real-time active transportation counts for key corridors.

## Requirements

This application requires a working Python environment capable of running Tensorflow.
Either Tensorflow GPU or Tensorflow CPU can be used, but the latter is likely too slow for real-time application.

## Installation

This application can run on environments both with and without GPUs.
However, if a GPU is not available,
it will likely run so slowly as to only be useful for testing/development.
If you are running it with a GPU,
you must already have the CUDA dynamic libraries installed and visible to the application.

The application comes with two conda environment specifications,
one for machines with a GPU, and one for machines without.
We assume you are installing the Python dependencies sugin `conda`.
This is not required, but if you are installing with another tool,
dependency manangement will likely be more difficult.
Instructions for installation and running are as follows:

1. Create a conda environment for the project. If running without GPUs, run
```bash
conda env create -f environment.yml
```
If running with GPUs, run
```bash
conda env create -f environment-gpu.yml
```
The given `environment.yml` files are known to work on at least some Linux, Windows, and Mac machines,
though you may want to choose a custom Tensorflow distribution depending on your deployment.
1. Activate the environment:
```bash
conda activate automated-walk-bike-counter
```
1. Install the application into the environment by running
```bash
pip install .
```
1. Launch the GUI by running `automated-walk-bike-counter`

## Running the application

#### GUI

In order to run the GUI, you must have an X server running on your machine.
This could be X11 on Mac OS or Linux, or Xming on Windows.
You may need to set your `DISPLAY` environment variable for the application to
find the running X server, i.e.
```bash
export DISPLAY=:0
```

Once you launch the application, you can run the algorithm with the following steps:
1. Select a file using the File menu.
1. Select which objects you want to track using the checkboxes in the left pane.
1. Select an area of interest using the Tools menu (optional)
1. Click the "Generate" button.

#### Command line usage

If you are running this application on a headless machine,
or automating it in some way, you will likely not want to use the GUI,
but instead drive it from the command line.

You can run it via the command line by using the `cli` argument:
```bash
automated-walk-bike-counter --cli True
```
Since this way of running the application doesn't have dialogs for selecting videos
and other options, you will need to provide more command line arguments.

For instance, to run the algorithm on a local video file with CSV exports at one minute intervals,
you should enter
```
automated-walk-bike-counter \
    --cli True \
    --file_name /path/to/your/video.mp4 \
    --input_type file \
    --save_periodic_counter True \
    --periodic_counter_interval 1
```

#### Configuration

The application is designed be be configured.
The full list of configurable settings can be viewed by running `automated-walk-bike-counter --help`.

All variables can be set in one of three ways:
1. Via command line argument.
1. Via environment variable (available environment variables are listed in the `--help` output).
1. Via config file.

If using a config file, you should add the settings in a `.ini` file,
and point the application at the file by running
```bash
automated-walk-bike-counter --config /path/to/your/config.ini
```
An example config file can be found in [`config.example.ini`](./config.example.ini).

## Development

In order to develop this project, you should make an editable dev install after creating your environment:
```bash
pip install -e .[develop]
```

You should then install the pre-commit hooks which are used to enforce code style
and lint for common errors:
```bash
pre-commit install
```
With these installed, all commits will get checked by the formatters and linters,
and the commit will fail if these checks fail.

Note: the first time that you make a commit with these hooks `pre-commit` will do some setup work.
This will take a few minutes. If you must, you can bypass the hooks by running `git commit --no-verify`.
