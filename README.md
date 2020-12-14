# Federated Kaplan-Meier Estimator for the FeatureCloud Platform

This FeatureCloud app is based on the [FeatureCloud Flask Template](https://github.com/FeatureCloud/flask_template).

### Usage

The app computes the survival function estimation of a csv file. The file should contain a time and event column. If a
category column is included, multiple survival functions will be estimated and a pairwise logrank test will be
performed.

To run the app, the input folder should contain a CSV/TSV/SASS file and a config.yml describing that file.
An example config.yml is included in this repository. 


### Technical Details

- This app overwrites the api.py and web.py of
  the [FeatureCloud Flask Template](https://github.com/FeatureCloud/flask_template).
- This app has no frontend
- In the requirements.txt are the project specific python requirements which will be installed in the docker image via
  Dockerfile
- The build.sh automatically builds the Federated Kaplan-Meier Estimator App with the image name fc_kaplan_meier
 