#  Write set of makefile to make your life easier
guard-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Required variable $* not set"; \
		exit 1; \
	fi

# Can not activate venv from makefile
#venv: guard-VENV_NAME
#	python3 -m venv ${VENV_NAME}

# venv activated can not persists after make command finished running
#activate:
#	@VENV_FOLDER="$$(ls -d venv_*)"; \
#	source $${VENV_FOLDER}/bin/activate; \
#	echo "$${VENV_FOLDER}"

install:
	python3 -m pip install --upgrade pip setuptools;
	python3 -m pip install -r requirements-no-versions.txt;

requirements:
	if [ -f requirements.txt ]; then \
  		cp requirements.txt requirements.txt.swp; \
	fi
	python3 -m pip list --format=freeze > requirements.txt;
	if [ -f requirements-no-versions.txt ]; then \
  		cp requirements-no-versions.txt requirements-no-versions.txt.swp; \
  	fi & \
  	python3 remove_package_versions.py;

remove_version:
	python3 remove_package_versions.py;