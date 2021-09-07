# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* waste_classification/*.py

black:
	@black scripts/* waste_classification/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr waste_classification-*.dist-info
	@rm -fr waste_classification.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)

# ----------------------------------
#      GOOGLE CLOUD PLATFORM
# ----------------------------------
PROJECT_ID="le-wagon-bootcamp-319614"

BUCKET_NAME="wagon-data-699-waste_classification"

REGION=europe-west2

PYTHON_VERSION=3.7
FRAMEWORK=tensorflow
RUNTIME_VERSION=1.15

PACKAGE_NAME=waste_classification
# FILENAME=gcp_trainer
FILENAME=trainer

JOB_NAME=waste_management_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

LOCAL_PATH_TRASHNET="/Users/izzy/code/MeylerL/waste_classification/raw_data/dataset-original"
LOCAL_PATH_TACO="/Users/izzy/code/MeylerL/waste_classification/raw_data/TACO/data/cat_folders"

BUCKET_FOLDER=waste_management_data

TRASHNET_BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH_TRASHNET})
TACO_BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH_TACO})

upload_trashnet_data:
	@gsutil cp -r ${LOCAL_PATH_TRASHNET} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${TRASHNET_BUCKET_FILE_NAME}

upload_TACO_data:
	@gsutil cp -r ${LOCAL_PATH_TACO} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${TACO_BUCKET_FILE_NAME}

test_cloud:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER}  \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
	      	--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--project le-wagon-bootcamp-319614 \
	 	--stream-logs
