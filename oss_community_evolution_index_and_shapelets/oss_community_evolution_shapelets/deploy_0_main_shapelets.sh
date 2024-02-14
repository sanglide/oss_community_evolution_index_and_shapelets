#!/bin/bash
TASK_SCRIPT="./server_scripts/run_main_shapelets.sh"
current_time=$(date "+%Y%m%d_%H%M%S")
REPO_NAME="feature_eng_oss_community_evolution_shapelets"
HOSTNAME="172.27.135.32"
USERNAME="wangliang"
REMOTE_DEPLOY_PATH="/home/wangliang/workspace/deploy/${REPO_NAME}/"
MKDIR_SCRIPT="mkdir -p ${REMOTE_DEPLOY_PATH}"
SCRIPT="cd ${REMOTE_DEPLOY_PATH}${REPO_NAME}_${current_time}_deploy/;pwd;chmod +x ${TASK_SCRIPT};bash -i ${TASK_SCRIPT}"

# pack the current code
mkdir -p ../deploy/${REPO_NAME}_${current_time}_deploy/
cp ./*.py ../deploy/${REPO_NAME}_${current_time}_deploy/
cp -r ./server_scripts/ ../deploy/${REPO_NAME}_${current_time}_deploy/
cd ../deploy/${REPO_NAME}_${current_time}_deploy/
dos2unix ${TASK_SCRIPT}

cd ..
echo 'Input password again to run: '${MKDIR_SCRIPT}
ssh -l ${USERNAME} ${HOSTNAME} "${MKDIR_SCRIPT}"
echo 'Input password to upload to server: '${HOSTNAME}
scp -r ./${REPO_NAME}_${current_time}_deploy/ ${USERNAME}@${HOSTNAME}:${REMOTE_DEPLOY_PATH}
echo 'Input password again to run: '${TASK_SCRIPT}
ssh -l ${USERNAME} ${HOSTNAME} "${SCRIPT}"