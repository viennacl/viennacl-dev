#/bin/bash

INPUT=$1
BUILDFOLDER=../build

# extract number of cores on the system
CORES=`grep -c ^processor /proc/cpuinfo`
echo "building with " $CORES "cores "

#./clean.sh
if [ "$INPUT" != "" ]; then
   if [ "$INPUT" == "help" ] ; then
      echo "usage: ./run_regression.sh [options]"
      echo ""
      echo "options:"
      echo "  help      prints this help message"
      echo "  submit    submits the result to online cdash service"
      exit -1
   elif [ "$INPUT" == "submit" ] ; then
      mkdir -p $BUILDFOLDER
      cd $BUILDFOLDER
      cmake ..
      # build and submit results to online cdash service
      make Nightly
      echo ""
      echo "regression result is available here:"
      echo "---------------------------------------------------------------------"
      echo "http://frudo.iue.tuwien.ac.at:50080/CDash/index.php?project=ViennaCL"
      echo "-------------------------------------------------------------------- "
      echo ""
   else
      echo ""
      echo "# Error - wrong option: \""$INPUT"\""
      echo ""
      echo "usage: ./run_regression.sh [options]"
      echo ""
      echo "options:"
      echo "  help      prints this help message"
      echo "  submit    submits the result to online cdash service"
      exit -1
   fi
else
   mkdir -p $BUILDFOLDER
   cd $BUILDFOLDER
   cmake ..
   # plain build without submit
   make
   make test
fi


