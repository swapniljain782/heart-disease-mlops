pipeline {
    agent { label 'built-in' }

    environment {
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                checkout scm
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
                // Using withEnv ensures these variables are injected into the shell session
                withEnv(['MLFLOW_TRACKING_URI=http://20.17.177.233:5000', 'MLFLOW_ALLOW_FILE_STORE=true']) {
                    echo "Starting model training..."
                    // Corrected to exactly three single quotes
                    sh '''
                        python3 -m venv venv
                        ./venv/bin/pip install mlflow scikit-learn pandas joblib matplotlib
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
