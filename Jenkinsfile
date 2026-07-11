pipeline {
    agent any

    // ADD THIS BLOCK: Tell kubectl exactly where to find the Minikube credentials
    environment {
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                checkout scm
                echo "Code checkout complete."
            }
        }

        stage('2. Build API Container') {
            steps {
                echo "Building the Docker image..."
                sh 'docker build -t heart-disease-api:latest .'
            }
        }

        stage('3. Deploy to Kubernetes') {
            steps {
                echo "Applying Kubernetes manifests..."
                // Now kubectl will use the config file and hit Minikube, not Jenkins
                sh 'kubectl apply -f k8s/'
                
                echo "Restarting the deployment..."
                sh 'kubectl rollout restart deployment heart-disease-api-deployment'
            }
        }

        stage('4. Verify Deployment') {
            steps {
                echo "Verifying pod status..."
                sh 'kubectl rollout status deployment/heart-disease-api-deployment --timeout=60s'
            }
        }
    }

    post {
        success {
            echo "✅ Deployment Successful! The new ML model is live."
        }
        failure {
            echo "❌ Deployment Failed. Please check the Jenkins console output to debug."
        }
    }
}
