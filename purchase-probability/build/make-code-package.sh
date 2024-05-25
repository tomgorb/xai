#!/usr/bin/env bash
set -e

CODE=purchase-probability

VERSION_NUMBER=2.0.1
TIMESTAMP=${BUILD_NUMBER:-0}

CONTAINER="build-deb-"$(echo $RANDOM % 100 + 1 | bc)

PACKAGE_PATH="/opt/${CODE}"
PACKAGE_ML=python/packages

docker build --tag $CODE \
             --file build/Dockerfile.code .

docker run -t --name $CONTAINER \
   --volume $(pwd)/python:/python \
   --volume $(pwd)/build:/build \
   $CODE \
   bash -c "pip3 install --upgrade pip \
   && pip3 install --upgrade wheel \
   && pip3 install --upgrade setuptools \
   && cd python \
   && echo $VERSION_NUMBER-$TIMESTAMP > version.txt \
   && python3 setup.py bdist_wheel \
   && cd .. \
   && python3 -m venv ${PACKAGE_PATH}/venv \
   && ${PACKAGE_PATH}/venv/bin/pip3 install --upgrade pip \
   && ${PACKAGE_PATH}/venv/bin/pip3 install --upgrade wheel \
   && ${PACKAGE_PATH}/venv/bin/pip3 install --upgrade setuptools \
   && ${PACKAGE_PATH}/venv/bin/pip3 install -r /python/requirements.txt \
   && ${PACKAGE_PATH}/venv/bin/pip3 install --no-deps python/dist/*.whl \
   && rm -f ${PACKAGE_ML}/*.whl\
   && mv python/dist/*.whl ${PACKAGE_ML}/ \
   && fpm \
      -s dir \
      -t deb \
      --deb-user yexp \
      --deb-group yexp \
      -n ${CODE} \
      -v ${VERSION_NUMBER} \
      --iteration ${TIMESTAMP} \
      --description 'CODE for RNN' \
      -p /build \
      ${PACKAGE_PATH}/=${PACKAGE_PATH} \
      /${PACKAGE_ML}/=${PACKAGE_PATH}/${PACKAGE_ML} \
      /python/main.py=${PACKAGE_PATH}/python/main.py \
      /python/queries.yaml=${PACKAGE_PATH}/python/queries.yaml \
    && rm python/version.txt \
    && rm -rf python/build \
    && rm -rf python/dist \
    && rm -rf python/*.egg-info \
    && rm -f ${PACKAGE_ML}/*.whl"

docker rm -f $CONTAINER

exit 0
