pipeline {
    agent { label 'built-in' }

    environment {
        KUBECONFIG = '/var/lib/jenkins/.kube/config'
    }

    stages {
        stage('1. Checkout Code') {
            steps {
                checkout scm
                // Ensure workspace folders exist and are writable
                sh '''
                    mkdir -p plots mlruns
                    chmod -R 777 plots mlruns
                '''
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
                // Set env variables to force proxy mode and use the workspace for local staging
                withEnv([
                    'MLFLOW_TRACKING_URI=http://20.17.177.233:5000',
                    'MLFLOW_HTTP_PROXY_ARTIFACTS=true',
                    'MLFLOW_ALLOW_FILE_STORE=false', // FORCE PROXY: Error if network fails
                    'MLFLOW_ARTIFACT_ROOT=' + env.WORKSPACE + '/mlruns'
                ]) {
                    echo "Starting model training..."
                    sh '''
                        # Ensure venv is ready
                        if [ ! -d "venv" ]; then
                            python3 -m venv venv
                            ./venv/bin/pip install --upgrade pip
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