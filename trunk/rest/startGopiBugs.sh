#!/bin/sh

SCRIPTDIR=`dirname "$0"`
cd "$SCRIPTDIR"

java -Djava.util.logging.config.file=conf/logging.properties -Xms512m -Xmx1024m -XX:ThreadStackSize=1024 -cp GopiBugs.jar alvs.main.ALVSClient
