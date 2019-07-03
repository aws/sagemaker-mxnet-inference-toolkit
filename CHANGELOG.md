# Changelog

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
