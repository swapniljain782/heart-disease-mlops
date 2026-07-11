pipeline {
    // Force the pipeline to execute ONLY on the main Jenkins host machine
    agent { 
        label 'built-in' 
    }

    environment {
        // Point Jenkins to its own dedicated configuration file
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
        // ADDED: Tell the environment where to find the MLflow server
        MLFLOW_TRACKING_URI = 'http://20.17.177.233:5000'
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

                echo "Injecting image into Minikube's internal registry..."
                // This saves the image and pipes it directly into the Minikube container
                sh 'docker save heart-disease-api:latest | docker exec -i minikube docker load'
            }
        }
       
        stage('Train Model') {
            environment {
                // Explicitly define here to ensure it hits the shell process
                MLFLOW_TRACKING_URI = 'http://20.17.177.233:5000'
            }
            steps {
                echo "Starting model training..."
                // Use 'sh' to verify it exists before running the script
                sh 'echo "Checking ENV: $MLFLOW_TRACKING_URI"'
                sh 'python3 src/train.py'
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
