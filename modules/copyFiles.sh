#!/bin/bash

SOURCEDIR=$(pwd)
DESTDIR=$SOURCEDIR/applications

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
		echo copying EXE $DIRECTORY/$FILE to $REMOTEDIRECTORY/$FILE
		cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
	    else
		if [ ${FILE: -3} == ".py" ]
		then
		    echo copying PY $DIRECTORY/$FILE to $REMOTEDIRECTORY/$FILE
		    cp $DIRECTORY/$FILE $REMOTEDIRECTORY/$FILE
		fi
            fi
	fi
    done
}

mkdir $DESTDIR
cd $SOURCEDIR
copy $SOURCEDIR $DESTDIR


