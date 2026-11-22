#!/bin/bash


data_dir="test-serving/"
S3_PATH=s3://govscape/eota-derivative-data/$data_dir*

# Remove all objects under the specified S3 path
s5cmd --endpoint-url https://data.source.coop rm "$S3_PATH"
