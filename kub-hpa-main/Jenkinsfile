pipeline {
    environment {
        DOCKERHUB_CRED = credentials("DockerSid")
        PORT = "8000" 
        VAULT_PASS = credentials("ansible_vault_pass")
    }
    agent any
    tools {nodejs "NODEJS"} 
    stages {
        stage("Git Clone") {
            steps {
                git credentialsId: 'GitHubSid', url: 'https://github.com/SiddharthChauhan303/SPE-Project-kub-hpa.git', branch: 'main'
            }
        }
        stage("Ansible-Kubernetes"){
            steps {
                sh '''
                echo "$VAULT_PASS" > /tmp/vault_pass.txt
                chmod 600 /tmp/vault_pass.txt
                sudo ansible-playbook -i inventory-k8 --vault-password-file /tmp/vault_pass.txt playbook-k8-new.yaml
                rm -f /tmp/vault_pass.txt
                '''
            }

        }
    }
}