#!/bin/bash

USERNAME=JENKINS

usage() {
  cat <<-EOF
      Usage: CI/CD Tools

      PARAMETRES:
      ===========
          build-app
          build-dag
          deploy

      OPTIONS:
      ========
          -h    Print this message

      EXAMPLES:
      =========

EOF
}

function delete_deb() {
  rm -f build/*.deb
}

function func_build_app() {
    ./build/make-code-package.sh
}

function func_build_dag() {
    ./build/make-dag-package.sh
}

function func_deploy() {
  for f in build/*.deb ; do
      echo $f;
  done
}

while getopts "h" arg
do
  case $arg in
    h)
    usage
    exit 0
    ;;
    ?)
    echo -e "\\033[31m Unknow argument \\033[0m"
    exit 1
    ;;
  esac
done

shift $((OPTIND-1))

case $1 in
  build-app)
  shift 1
  delete_deb
  func_build_app "$@"
  ;;
  build-dag)
  shift 1
  delete_deb
  func_build_dag "$@"
  ;;
  deploy)
  shift 1
  func_deploy "$@"
  ;;
  "")
  echo -e "\033[33m No Options \033[0m"
  exit 0
  ;;
  *)
  echo -e "\\033[31m Error \\033[0m No Such Option" 1>&2
  exit 1
  ;;
esac
