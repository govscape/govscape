import boto3

AMI_ID = 'ami-0321559df76fad329'
INSTANCE_TYPE = 'g4dn.4xlarge'  # Change as needed
KEY_NAME = 'kyle-desktop'  # Replace with your EC2 key pair name
IAM_INSTANCE_PROFILE = {'Name': 'GovScapeServerEC2Role'}
NUM_SERVERS = 2
NUM_PAGES_TO_PROCESS = 2

ec2 = boto3.client('ec2')

user_data_template = '''
#!/bin/bash
cd /home/ubuntu
git clone https://github.com/bcglee/govscape.git || true
cd govscape
git stash
git pull
poetry lock
poetry install
poetry run python scripts/python_helpers/s3_embedding_pipeline.py --num_pages_to_process {num_pages} --bucket_name 'bcgl-public-bucket' --pdf_dir 'archive/2020/PDFs/' --data_dir "dev-serving/" --model_type 'BGE' --num_servers {num_servers} --server_id {server_id}
'''

for i in range(NUM_SERVERS):
    user_data = user_data_template.format(num_pages=NUM_PAGES_TO_PROCESS, num_servers=NUM_SERVERS, server_id=i)
    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        IamInstanceProfile=IAM_INSTANCE_PROFILE,
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'govscape-server-{i}'}
                ]
            }
        ]
    )
    print(f"Started instance {i}: {response['Instances'][0]['InstanceId']}")
