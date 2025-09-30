import boto3

AMI_ID = 'ami-0321559df76fad329'
INSTANCE_TYPE = 'g4dn.4xlarge'  # Change as needed
KEY_NAME = 'kyle-desktop'  # Replace with your EC2 key pair name
IAM_INSTANCE_PROFILE = {'Name': 'GovScapeServerEC2Role'}
NUM_SERVERS = 30
SECURITY_GROUPS = [{'GroupId': 'sg-0e4b8310618ef3b7a'}]  # Replace with your security group ID
NUM_PAGES_TO_PROCESS = 10000000

ec2 = boto3.client('ec2', region_name='us-east-2')

user_data_template = '''#!/bin/bash
sudo -u ubuntu bash -c "
cd /home/ubuntu/govscape && \
git stash >> /home/ubuntu/govscape/log.txt && \
git pull >> /home/ubuntu/govscape/log.txt && \
rm /home/ubuntu/govscape/progress.json || true && \
/home/ubuntu/.local/bin/poetry lock >> /home/ubuntu/govscape/log.txt && \
/home/ubuntu/.local/bin/poetry install >> /home/ubuntu/govscape/log.txt && \
/home/ubuntu/.local/bin/poetry run python scripts/python_helpers/s3_embedding_pipeline.py \
    --num_pages_to_process {num_pages} \
    --bucket_name 'bcgl-public-bucket' \
    --pdf_dir 'archive/2020/PDFs/' \
    --data_dir 'dev-serving/' \
    --model_type 'BGE' \
    --num_servers {num_servers} \
    --server_id {server_id} >> /home/ubuntu/govscape/log.txt
"
'''

for i in range(NUM_SERVERS):
    user_data = user_data_template.format(num_pages=NUM_PAGES_TO_PROCESS, num_servers=NUM_SERVERS, server_id=i)
    response = ec2.run_instances(
        ImageId=AMI_ID,
        InstanceType=INSTANCE_TYPE,
        KeyName=KEY_NAME,
        IamInstanceProfile=IAM_INSTANCE_PROFILE,
        SecurityGroupIds=[sg['GroupId'] for sg in SECURITY_GROUPS],
        MinCount=1,
        MaxCount=1,
        UserData=user_data,
        TagSpecifications=[
            {
                'ResourceType': 'instance',
                'Tags': [
                    {'Key': 'Name', 'Value': f'embedding-server-{i}'}
                ]
            }
        ]
    )
    print(f"Started instance {i}: {response['Instances'][0]['InstanceId']}")
