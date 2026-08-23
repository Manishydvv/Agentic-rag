# AWS Fargate Deployment Guide

This document outlines the deployment strategy for the Agentic RAG API on AWS ECS Fargate, including the exact CLI commands used and a breakdown of the architectural choices.

---

## 🏗️ Architectural Strategies

There are two primary ways to deploy this application to AWS depending on your budget and scale.

### 1. The "Lean Startup" Architecture (Currently Used)
For testing, prototyping, and low-traffic personal projects, we use a highly cost-optimized approach.

- **Location:** The Fargate container is placed directly into a **Public Subnet**.
- **No Load Balancer:** We skip the Application Load Balancer (ALB) to save ~$20/month.
- **No NAT Gateway:** We skip the NAT Gateway to save ~$32/month.
- **Access:** AWS assigns a **Public IP address** directly to the Fargate container (`assignPublicIp=ENABLED`). You access the API directly via this IP (e.g., `http://54.12.34.56:8000`).
- **Security:** Security is strictly enforced by a **Security Group** attached directly to the container, which brutally blocks all traffic except for Port 8000.

### 2. The "Enterprise Production" Architecture (Future Upgrade)
When the application scales to serve real customers and requires a custom domain name (e.g., `api.mycompany.com`) and SSL (HTTPS), the architecture must be upgraded.

- **Public & Private Subnets:** The VPC is divided into Public and Private subnets.
- **Application Load Balancer (ALB):** Placed in the Public Subnet. It acts as the "Receptionist," receiving web traffic, handling SSL encryption, and distributing traffic across multiple Fargate containers.
- **Fargate Containers:** Placed in the **Private Subnet**. They are completely hidden from the internet. They only accept traffic directly from the ALB's Security Group.
- **NAT Gateway:** Placed in the Public Subnet. Since the Fargate containers are hidden in the Private Subnet, they must route their outbound requests (like calling the Groq LLM API or Qdrant Cloud) through the NAT Gateway to reach the internet securely.

---

## 🚀 Deployment Commands (Lean Startup)

To deploy the current architecture using the AWS CLI, execute the following commands. 

*(Note: Ensure you have run `aws configure` with an IAM user that has `AmazonEC2ContainerRegistryFullAccess` and `AmazonECS_FullAccess` before proceeding).*

### Step 1: Create Repositories and CloudWatch Logs
```bash
aws ecr create-repository --repository-name agentic-rag-api --region us-east-1

aws logs create-log-group --log-group-name /ecs/agentic-rag-api --region us-east-1
```

### Step 2: Build and Push Docker Image
```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# Build the Docker image natively
docker build -t agentic-rag-api .

# Tag the image for your ECR repository
docker tag agentic-rag-api:latest <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/agentic-rag-api:latest

# Push the image to AWS
docker push <YOUR_AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/agentic-rag-api:latest
```

### Step 3: Create ECS Cluster and Task Definition
*(Ensure your `task-def.json` file is present in the root directory and contains your database URLs and API keys).*
```bash
aws ecs create-cluster --cluster-name agentic-rag-cluster

aws ecs register-task-definition --cli-input-json file://task-def.json
```

### Step 4: Launch the Fargate Service
```bash
aws ecs create-service \
    --cluster agentic-rag-cluster \
    --service-name agentic-rag-service \
    --task-definition agentic-rag-task \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[<YOUR_SUBNET_ID>],securityGroups=[<YOUR_SECURITY_GROUP_ID>],assignPublicIp=ENABLED}"
```

### Next Steps:
Once the service is running, log into the AWS Console, navigate to **ECS > Clusters > agentic-rag-cluster > Tasks**, click on the running task, and copy the **Public IP**. You can access your API docs at `http://<PUBLIC_IP>:8000/docs`.

To automate this process for future code updates, configure your GitHub Repository Secrets and push to the `main` branch to trigger the `.github/workflows/deploy.yml` CI/CD pipeline.
