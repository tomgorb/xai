#!/usr/bin/env bash

DAG=dag-purchase-probability

VERSION_NUMBER=1.0.1
TIMESTAMP=${BUILD_NUMBER:-0}

CONTAINER="build-deb-"$(echo $RANDOM % 100 + 1 | bc)

PACKAGE_PATH="/opt/airflow/dags/${DAG}"

docker build --tag $DAG \
             --file build/Dockerfile.dag .

docker run -t --name $CONTAINER \
    --volume $(pwd)/dags:/dags \
    --volume $(pwd)/build:/build \
    $DAG \
    bash -c "fpm \
              -s dir \
              -t deb \
              --deb-user yexp \
              --deb-group yexp \
              -n ${DAG} \
              -v ${VERSION_NUMBER} \
              --iteration ${TIMESTAMP} \
              --description 'DAG for RNN' \
              -p /build \
              /dags/${DAG}.py=${PACKAGE_PATH}/${DAG}.py"

docker rm -f $CONTAINER

exit 0
