pipeline {
    agent { label 'built-in' }

    environment {
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                checkout scm
                // Force the Jenkins user to take ownership of the workspace
                sh 'chown -R jenkins:jenkins .'
                sh 'chmod -R 755 .'
            }
        }

        stage('2. Build API Container') {
            steps {
                sh 'docker build -t heart-disease-api:latest .'
                sh 'docker save heart-disease-api:latest | docker exec -i minikube docker load'
            }
        }

        stage('3. Train Model') {
            steps {
                script {
                    echo "Current Workspace: ${env.WORKSPACE}"
                    sh 'ls -ld .'
                   }
                // Using withEnv ensures these variables are injected into the shell session
                withEnv(['MLFLOW_TRACKING_URI=http://20.17.177.233:5000', 'MLFLOW_ALLOW_FILE_STORE=true', 'MLFLOW_HTTP_PROXY_ARTIFACTS=true']) {
                    echo "Starting model training..."
                    // Corrected to exactly three single quotes
                    sh '''
                        # Create venv if it doesn't exist
                        if [ ! -d "venv" ]; then
                            python3 -m venv venv
                            ./venv/bin/pip install mlflow scikit-learn pandas joblib matplotlib
                        fi
                        # Run the updated training script
                        ./venv/bin/python src/train.py
                    '''
                }
            }
        }

        stage('4. Deploy to Kubernetes') {
            steps {
                sh 'kubectl apply -f k8s/'
                sh 'kubectl rollout restart deployment heart-disease-api-deployment'
            }
        }

        stage('5. Verify Deployment') {
            steps {
                sh 'kubectl rollout status deployment/heart-disease-api-deployment --timeout=60s'
            }
        }
    }

    post {
        success {
            echo "✅ Deployment Successful!"
        }
        failure {
            echo "❌ Deployment Failed."
        }
    }
}
