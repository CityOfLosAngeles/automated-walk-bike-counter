# Automated Walk Bike Counter

## About

The City currently does bicycle and pedestrian counts via having a person manually the number of cyclists and pedestrians that go through an intersection via a video capture.

However, thanks to advances in computer vision we can now automate that,
allowing us to constantly count the number of pedestrians and cyclists,
rather than sampling a (possibly not representative) time and location.

This project is a proof of concept of how we could count the number folks moving through our streets.

## Sponsors

Toyota Mobility Foundation - William Chernicoff

ITA - Jeanne Holm

DOT - Seleta Reynolds, Marcel Porras

## Partners

CSU LA, Dr. Mohammad Pourhomayoun

## City Team

Hunter Owens

Ian Rose

Janna Smith

## Goals

Allow us to know real-time active transportation counts for key corridors.

## Deliverables

Using computer vision, take data from video feeds and produce bicycle and pedestrian counts,
publishing them using open data standards so that they are ready-to-be used for policymaking.

## Data Sources

Model weights for the computer vision algorithm may be found at `s3://automated-walk-bike-counter`.

## Requirements

This application requires a working Python environment capable of running Tensorflow.
Either Tensorflow GPU or Tensorflow CPU can be used,
but the latter is likely too slow for real-time application.

## Installation

1. Create a conda environment for the project:
```bash
conda env create -f environment.yml
```
The given `environment.yml` is known to work on at least some Linux, Windows, and Mac machines,
though you may want to choose a custom Tensorflow distribution depending on your deployment.

2. Install the project
```bash
pip install .
```
3. Launch the GUI by running `automated-walk-bike-counter`

## Configuration

The application is designed be be configured.
An example config file can be found in `config.example.ini`.
Configuration can also be passed in via environment variables or command line options.

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
