# Changelog

## v1.4.0 (2020-05-12)

### Features

 * Python 3.7 support

### Testing and Release Infrastructure

 * remove unused scripts.

## v1.3.3.post0 (2020-04-30)

### Testing and Release Infrastructure

 * use tox in buildspecs

## v1.3.3 (2020-04-28)

### Bug Fixes and Other Changes

 * Improve error message in handler_service.py

## v1.3.2 (2020-04-01)

### Bug Fixes and Other Changes

 * upgrade inference-toolkit version

## v1.3.1 (2020-04-01)

### Bug Fixes and Other Changes

 * add model_dir to python path at service initialization
 * add gluoncv

## v1.3.0 (2020-03-30)

### Features

 * install mxnet-inference toolkit from PyPI.

## v1.2.6.post0 (2020-03-24)

### Testing and Release Infrastructure

 * refactor toolkit tests.

## v1.2.6 (2020-03-11)

### Bug Fixes and Other Changes

 * Test MME hosting with MultiDataModel

## v1.2.5 (2020-03-09)

### Bug Fixes and Other Changes

 * Upgrade the version of sagemaker-inference

## v1.2.4 (2020-03-04)

### Bug Fixes and Other Changes

 * Modify HandlerService to use context model_dir

## v1.2.3 (2020-02-20)

### Bug Fixes and Other Changes

 * copy all tests to test-toolkit folder.

## v1.2.2 (2020-02-19)

### Bug Fixes and Other Changes

 * change remove multi-model label from dockerfiles

## v1.2.1 (2020-02-17)

### Bug Fixes and Other Changes

 * update: Update license URL

## v1.2.0 (2020-02-12)

### Features

 * Add release to PyPI. Change package name to sagemaker-mxnet-inference.

### Bug Fixes and Other Changes

 * Add GluonNLP
 * Update AWS-MXNet version to 1.6.0 - official release of 1.6
 * Update build artifacts
 * Revert "change: install python-dateutil explicitly for botocore (#81)"
 * pin in setuptools in py2 containers
 * make build context the directory with the given dockerfile
 * update copyright year in license header
 * add comments about sagemaker-inference version
 * release 1.6.0 Dockerfiles
 * install python-dateutil explicitly for botocore
 * use regional endpoint for STS in builds
 * Update toolkit version

### Testing and Release Infrastructure

 * properly fail build if has-matching-changes fails
 * properly fail build if has-matching-changes fails

## v1.1.5 (2019-10-22)

### Bug fixes and other changes

 * update instance type region availability

## v1.1.4 (2019-09-26)

### Bug fixes and other changes

 * build context in release build
 * make consistent build context for py versions
 * Revert "Update build context on 1.4.1 EI dockerfiles (#64)"
 * a typo in build_all script
 * Update build context on 1.4.1 EI dockerfiles
 * correct typo in buildspec
 * enable PR build
 * separate pip upgrade from other installs
 * New py2 dlc dockerfiles
 * New py3 dlc dockerfiles

## v1.1.3.post0 (2019-08-29)

### Documentation changes

 * Update the build instructions to match reality.

## v1.1.3 (2019-08-28)

### Bug fixes and other changes

 * retry mms until it's ready

## v1.1.2 (2019-08-17)

### Bug fixes and other changes

 * split cpu and gpu tests for deployments in buildspec-release.yml
 * add missing placeholder in test for non-eia test command.

## v1.1.1 (2019-08-15)

### Bug fixes and other changes

 * fix flake8
 * update no-p2 regions and no-p3 regions.
 * Skipping EIA test if accelerator type is None.

## v1.1.0 (2019-07-18)

### Features

 * add MXNet 1.4.1 Dockerfiles

### Bug fixes and other changes

 * use Python 2 build logic for EI images during release
 * add missing files needed for building MXNet 1.4.1 Python 2 images
 * configure flake8 to ignore the docker/ directory

## v1.0.6 (2019-07-03)

### Bug fixes and other changes

 * add retries to all integ tests commands in buildspec-release.yml

## v1.0.5 (2019-07-03)

### Bug fixes and other changes

 * fix account number for EI deployment test in buildspec-release.yml

## v1.0.4 (2019-07-01)

### Bug fixes and other changes

 * remove unnecessary pytest marks
 * add retries to remote integ tests in buildspec-release.yml
 * update tests to except pytest's ExceptionInfo object
 * parametrize Python version and processor type in integ tests

## v1.0.3 (2019-06-28)

### Bug fixes and other changes

 * fix account number in deployment test command

## v1.0.2 (2019-06-28)

### Bug fixes and other changes

 * remove nonexistent EI GPU images from buildspec-release.yml

## v1.0.1 (2019-06-27)

### Bug fixes and other changes

 * add release buildspec
 * Fix SageMaker integration tests
 * skip GPU test in regions with limited p2s or p3s
 * Add link to SageMaker integ test setup requirements to README
 * Add SageMaker Elastic Inference test

## v1.0.0

Initial commit
