#!/bin/bash

SOURCEDIR=$(pwd)
DESTDIR=$SOURCEDIR/../v2.5.0/applications

copy() {
    local DIRECTORY=$1
    local REMOTEDIRECTORY=$2

    for FILE in `ls $DIRECTORY`
    do
        if [ -d $DIRECTORY/$FILE ]
        then
	    if [ $FILE != 'applications' ]
		then
		   echo creating DIR $REMOTEDIRECTORY/$FILE 
		   mkdir $REMOTEDIRECTORY/$FILE
		   copy $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
	    fi
        else
            if [ -x $DIRECTORY/$FILE ]
            then
		#echo copying EXE $DIRECTORY/$FILE to $REMOTEDIRECTORY/$FILE
		echo copying EXE $DIRECTORY/$FILE
		cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
	    else
		if [ ${FILE: -3} == ".py" ]
		then
		    echo copying PY $DIRECTORY/$FILE 
		    cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
		fi
		if [ ${FILE: -3} == "hdf" ]
		then
		    echo copying HDF5 $DIRECTORY/$FILE 
		    cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
		fi
		if [ ${FILE: -3} == "son" ]
		then
		    echo copying JSON $DIRECTORY/$FILE 
		    cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
		fi
            fi
	fi
    done
}

#rm -fr $DESTDIR
#mkdir $DESTDIR
cd $SOURCEDIR
copy $SOURCEDIR $DESTDIR

chmod -R 'a+rx' $DESTDIR
echo FINISHED DEST_DIR $DESTDIR

