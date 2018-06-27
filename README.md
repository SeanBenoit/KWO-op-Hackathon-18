# KWO-op-Hackathon-18

## Pre-Setup
Create directory /data
cd into /data
clone this repo put its contents in /data

## Setup Cluster
Follow instructions here: https://github.com/brunocfnba/docker-spark-cluster
At docker create step use -v arg as /data:/data

## Jobs
Edit Code under src dir

Run `mvn clean package` in root to compile

Run Job: `docker run --rm -it --link master:master --volumes-from spark-datastore brunocf/spark-submit spark-submit --master spark://172.17.0.2:7077 --class hack.train.trainModel /data/target/assignments-1.0.jar --input /data/test.txt --output /data/testOut`

Run Example LinerSGD: `docker run --rm -it --link master:master --volumes-from spark-datastore brunocf/spark-submit spark-submit --master spark://172.17.0.2:7077 --class hack.train.LinearRegressionWithSGDExample /data/target/assignments-1.0.jar`
