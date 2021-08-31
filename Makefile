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
FILENAME=gcp_trainer

JOB_NAME=waste_management_training_pipeline_$(shell date +'%Y%m%d_%H%M%S')

set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}

#RENAME WHEN WE'VE GOT THE DATAPATH
LOCAL_PATH_TRASHNET="XXX"
LOCAL_PATH_TACO="XXX"

BUCKET_FOLDER=waste_management_data

BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})

upload_trashnet_data:
	@gsutil cp ${LOCAL_PATH_TASHNET} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}

upload_TACO_data:
	@gsutil cp ${LOCAL_PATH_TACO} gs://${BUCKET_NAME}/${BUCKET_FOLDER}/${BUCKET_FILE_NAME}
