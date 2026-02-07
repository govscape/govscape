#!/bin/bash


data_dir="test-serving/"
S3_PATH=s3://bcgl-public-bucket/$data_dir*

# Remove all objects under the specified S3 path
s5cmd rm "$S3_PATH"
