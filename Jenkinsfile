pipeline {
    // 'agent any' tells Jenkins to execute this pipeline on any available worker node
    agent any

    stages {
        stage('1. Checkout Code') {
            steps {
                // Pulls the latest code from the branch specified in the Jenkins UI
                checkout scm
                echo "Code checkout complete."
            }
        }

        stage('2. Build API Container') {
            steps {
                // Builds the Docker image for the FastAPI app and ML model
                echo "Building the Docker image..."
                sh 'docker build -t heart-disease-api:latest .'
            }
        }

        stage('3. Deploy to Kubernetes') {
            steps {
                // Applies your Deployment and Service manifests to the cluster
                echo "Applying Kubernetes manifests..."
                sh 'kubectl apply -f k8s/'
                
                // Forces a rolling restart so the pods pull the newly built image
                echo "Restarting the deployment..."
                sh 'kubectl rollout restart deployment heart-disease-api-deployment'
            }
        }

        stage('4. Verify Deployment') {
            steps {
                // Halts the pipeline until Kubernetes confirms the new pods are running successfully
                echo "Verifying pod status..."
                sh 'kubectl rollout status deployment/heart-disease-api-deployment --timeout=60s'
            }
        }
    }

    // The post block executes after all stages complete, regardless of success or failure
    post {
        success {
            echo "✅ Deployment Successful! The new ML model is live."
        }
        failure {
            echo "❌ Deployment Failed. Please check the Jenkins console output to debug."
        }
    }
}
