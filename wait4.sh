#!/bin/sh
$1 > test.out 2>&1 3>&1 &
PROCESS="$!"
while :
do
  RESULT=`ps -p ${PROCESS} -o comm=`
  if [ -z "${RESULT}" ]; then
    wait ${PROCESS}; exit $?
  else
    echo "-"; sleep 10
  fi
done
RETURNCODE="$?"
cat test.out
exit $RETURNCODE
