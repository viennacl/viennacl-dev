#!/usr/bin/env bash

if [ $# -ne 2 ]; then
  cat >&2 << EOF
Usage: $0 <build-name> <dashboard-type>

<build-name> should be of the format "<OS-ID>-<COMPILER-ID>".
<dashboard-type> should be one of Nightly, Experimental or Continuous.

EOF
  exit 1
fi

# run in a clean shell
if [ -z "$IN_CLEAN_SHELL" ]; then
  env - IN_CLEAN_SHELL=yes \
        HOME=$HOME \
        PATH=/bin:/usr/bin \
        bash --norc $0 $@
  exit $?
fi

LOGDIR=$PWD
cd ${0%/*}/.. || exit 1

build_name=$1
dashboard_type=$2
start_time=`date "+%H:%M:00 %Z"`
script_arg="CTEST_BUILD_NAME=$build_name"
script_arg="$script_arg;DASHBOARD_TYPE=$dashboard_type"
script_arg="$script_arg;CTEST_NIGHTLY_START_TIME=$start_time"

set -e
set -x

ctest -VV -S cmake/dashboard_script.cmake,"$script_arg" > \
  $LOGDIR/$build_name-$dashboard_type.log 2>&1
