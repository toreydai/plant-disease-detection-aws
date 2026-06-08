#!/usr/bin/env python3
"""完整端到端流水线：Training Job（四模型）→ 打包 → 更新 Endpoint。

用法：
    python3 run_pipeline.py              # 四模型全跑
    python3 run_pipeline.py --yolo-only  # 仅 YOLO（快，约 15-20min）
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

import boto3

# ── 配置 ───────────────────────────────────────────────────────────────
# 从环境变量读取，未设置时自动探测
REGION  = os.environ.get('AWS_REGION', 'us-east-1')
_sts    = boto3.client('sts', region_name=REGION)
ACCOUNT = _sts.get_caller_identity()['Account']
BUCKET  = os.environ.get('SAGEMAKER_BUCKET', f'sagemaker-{REGION}-{ACCOUNT}')

# SageMaker 执行角色：优先环境变量，其次自动查找账号下第一个 ExecutionRole
def _find_role():
    if r := os.environ.get('SAGEMAKER_ROLE_ARN'):
        return r
    iam   = boto3.client('iam', region_name=REGION)
    roles = iam.list_roles(PathPrefix='/service-role/')['Roles']
    for r in roles:
        if 'SageMaker' in r['RoleName'] and 'ExecutionRole' in r['RoleName']:
            return r['Arn']
    sys.exit('错误：未找到 SageMaker ExecutionRole，请设置环境变量 SAGEMAKER_ROLE_ARN')

ROLE_ARN        = _find_role()
PREFIX          = 'potato-demo'
ENDPOINT_NAME   = 'potato-disease-demo'
INSTANCE_TRAIN  = 'ml.g5.2xlarge'
INSTANCE_INFER  = 'ml.m5.large'
IMAGE_INFER     = f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-inference:2.3.0-cpu-py311-ubuntu20.04-sagemaker'

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


def ts():
    return datetime.datetime.now().strftime('%H:%M:%S')


def log(msg):
    print(f'[{ts()}] {msg}', flush=True)


# ── Step 1: 打包训练源代码并上传到 S3 ─────────────────────────────────
def upload_source(s3):
    log('Step 1: 打包 sagemaker/ 源代码 → S3')
    src_dir = os.path.join(PROJECT_DIR, 'sagemaker')

    tmpfile = os.path.join(PROJECT_DIR, '_sourcedir.tar.gz')
    with tarfile.open(tmpfile, 'w:gz') as tf:
        tf.add(os.path.join(src_dir, 'train_job.py'), arcname='train_job.py')
        tf.add(os.path.join(src_dir, 'requirements.txt'), arcname='requirements.txt')
    log(f'  打包完成: {os.path.getsize(tmpfile)} bytes')

    s3_key = f'{PREFIX}/source/sourcedir_v2.tar.gz'
    s3.upload_file(tmpfile, BUCKET, s3_key)
    os.remove(tmpfile)
    source_uri = f's3://{BUCKET}/{s3_key}'
    log(f'  上传完成: {source_uri}')
    return source_uri


# ── Step 2: 提交 Training Job ──────────────────────────────────────────
def submit_training_job(sm, source_uri, models_str):
    job_name = f'potato-4models-{int(time.time())}'
    log(f'Step 2: 提交 Training Job: {job_name}')
    log(f'  实例: {INSTANCE_TRAIN}  模型: {models_str}')

    response = sm.create_training_job(
        TrainingJobName=job_name,
        RoleArn=ROLE_ARN,
        AlgorithmSpecification={
            'TrainingInputMode': 'File',
            'TrainingImage': f'763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.3.0-gpu-py311-cu121-ubuntu20.04-sagemaker',
        },
        HyperParameters={
            'sagemaker_program':          'train_job.py',
            'sagemaker_submit_directory': source_uri,
            'sagemaker_region':           REGION,
            'models':                     models_str,
            'yolo-epochs':                '300',
            'resnet-epochs':              '25',
            'mobile-epochs':              '100',
            'cnn-epochs':                 '100',
            'img-size':                   '224',
            'batch-size':                 '16',
            'early-stop-patience':        '8',
        },
        InputDataConfig=[{
            'ChannelName': 'data',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': f's3://{BUCKET}/{PREFIX}/data/',
                    'S3DataDistributionType': 'FullyReplicated',
                }
            },
        }],
        OutputDataConfig={
            'S3OutputPath': f's3://{BUCKET}/{PREFIX}/output/',
        },
        ResourceConfig={
            'InstanceType': INSTANCE_TRAIN,
            'InstanceCount': 1,
            'VolumeSizeInGB': 30,
        },
        StoppingCondition={'MaxRuntimeInSeconds': 7200},
    )
    log(f'  已提交: {response["TrainingJobArn"]}')
    return job_name


# ── Step 3: 等待 Training Job 完成 ─────────────────────────────────────
def wait_for_training(sm, job_name):
    log(f'Step 3: 等待训练完成（每 60s 刷新一次）...')
    last_status = None
    while True:
        resp = sm.describe_training_job(TrainingJobName=job_name)
        status = resp['TrainingJobStatus']
        if status != last_status:
            log(f'  状态: {status}')
            last_status = status
        if status == 'Completed':
            model_uri = resp['ModelArtifacts']['S3ModelArtifacts']
            bill_secs = resp.get('BillableTimeInSeconds', 0)
            log(f'  训练完成！计费 {bill_secs}s ({bill_secs/60:.1f}min)')
            log(f'  模型输出: {model_uri}')
            return model_uri
        elif status in ('Failed', 'Stopped'):
            reason = resp.get('FailureReason', '无详情')
            log(f'  ERROR: Training Job {status}: {reason}')
            sys.exit(1)
        time.sleep(60)


# ── Step 4: 下载训练输出，重新打包含 inference 代码的 deploy 包 ─────────
def repack_model(s3, model_uri):
    log('Step 4: 下载训练产出，打包 deploy model.tar.gz（含 inference.py）')

    workdir = os.path.join(PROJECT_DIR, '_deploy_tmp')
    os.makedirs(workdir, exist_ok=True)

    # 下载 Training Job 的 model.tar.gz
    bucket, key = model_uri.replace('s3://', '').split('/', 1)
    train_tar = os.path.join(workdir, 'train_model.tar.gz')
    s3.download_file(bucket, key, train_tar)
    log(f'  下载完成: {os.path.getsize(train_tar)/1024:.0f} KB')

    # 解压取 best.pt
    extract_dir = os.path.join(workdir, 'extracted')
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(train_tar) as tf:
        members = tf.getmembers()
        log(f'  训练输出文件: {[m.name for m in members]}')
        tf.extractall(extract_dir)

    best_pt = os.path.join(extract_dir, 'best.pt')
    if not os.path.exists(best_pt):
        log('  ERROR: best.pt 不在训练输出中')
        sys.exit(1)
    log(f'  best.pt: {os.path.getsize(best_pt)/1024:.0f} KB')

    # 检查 comparison.json
    cmp_json = os.path.join(extract_dir, 'comparison.json')
    if os.path.exists(cmp_json):
        with open(cmp_json) as f:
            cmp = json.load(f)
        log('  === 四模型对比结果 ===')
        for model_name, metrics in sorted(cmp.items(), key=lambda x: -x[1].get('val_acc', 0)):
            log(f'    {model_name}: val_acc={metrics["val_acc"]:.4f}  '
                f'time={metrics["train_min"]:.1f}min  params={metrics["params_m"]:.2f}M')
    else:
        log('  注意：comparison.json 未找到（可能只训了 YOLO）')

    # 打包 deploy 包：best.pt + code/
    deploy_tar = os.path.join(workdir, 'model.tar.gz')
    sm_dir = os.path.join(PROJECT_DIR, 'sagemaker')
    with tarfile.open(deploy_tar, 'w:gz') as tf:
        tf.add(best_pt, arcname='best.pt')
        tf.add(os.path.join(sm_dir, 'inference.py'),             arcname='code/inference.py')
        tf.add(os.path.join(sm_dir, 'requirements.txt'), arcname='code/requirements.txt')
    log(f'  打包完成: {os.path.getsize(deploy_tar)/1024:.0f} KB')

    # 上传到 S3
    deploy_key = f'{PREFIX}/deploy/model_v2.tar.gz'
    s3.upload_file(deploy_tar, BUCKET, deploy_key)
    deploy_uri = f's3://{BUCKET}/{deploy_key}'
    log(f'  上传完成: {deploy_uri}')

    # 清理
    shutil.rmtree(workdir)
    return deploy_uri


# ── Step 5: 创建新 SageMaker Model ────────────────────────────────────
def create_model(sm, deploy_uri):
    ts_suffix = int(time.time())
    model_name = f'potato-model-4m-{ts_suffix}'
    log(f'Step 5: 创建 SageMaker Model: {model_name}')

    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=ROLE_ARN,
        PrimaryContainer={
            'Image': IMAGE_INFER,
            'ModelDataUrl': deploy_uri,
            'Environment': {
                'SAGEMAKER_PROGRAM':          'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/model/code',
                'SAGEMAKER_CONTAINER_LOG_LEVEL': '20',
                'SAGEMAKER_REGION':           REGION,
            },
        },
    )
    log(f'  Model 创建完成')
    return model_name


# ── Step 6: 创建新 EndpointConfig ─────────────────────────────────────
def create_endpoint_config(sm, model_name):
    ts_suffix = int(time.time())
    cfg_name = f'potato-cfg-4m-{ts_suffix}'
    log(f'Step 6: 创建 EndpointConfig: {cfg_name}')

    sm.create_endpoint_config(
        EndpointConfigName=cfg_name,
        ProductionVariants=[{
            'VariantName':           'AllTraffic',
            'ModelName':             model_name,
            'InitialInstanceCount':  1,
            'InstanceType':          INSTANCE_INFER,
            'InitialVariantWeight':  1.0,
        }],
    )
    log(f'  EndpointConfig 创建完成')
    return cfg_name


# ── Step 7: 更新 Endpoint ─────────────────────────────────────────────
def update_endpoint(sm, cfg_name):
    log(f'Step 7: 更新 Endpoint: {ENDPOINT_NAME} → {cfg_name}')

    # 检查 Endpoint 是否存在
    try:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        current_status = resp['EndpointStatus']
        log(f'  当前状态: {current_status}，执行 update_endpoint')
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=cfg_name,
        )
    except sm.exceptions.ClientError:
        log(f'  Endpoint 不存在，执行 create_endpoint')
        sm.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=cfg_name,
        )

    # 等待 InService
    log(f'  等待 InService（每 30s 刷新）...')
    last_status = None
    while True:
        resp = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        status = resp['EndpointStatus']
        if status != last_status:
            log(f'  Endpoint 状态: {status}')
            last_status = status
        if status == 'InService':
            log(f'  ✓ Endpoint InService')
            return
        elif status in ('Failed', 'OutOfService'):
            reason = resp.get('FailureReason', '无详情')
            log(f'  ERROR: Endpoint {status}: {reason}')
            sys.exit(1)
        time.sleep(30)


# ── Step 8: 冒烟测试 ──────────────────────────────────────────────────
def smoke_test():
    log('Step 8: 冒烟测试（调用 Endpoint）')
    import base64
    import glob

    # 找一张验证集图片
    val_dir = os.path.join(PROJECT_DIR, 'notebooks', 'data', 'val')
    images = (glob.glob(f'{val_dir}/**/*.jpg', recursive=True) +
              glob.glob(f'{val_dir}/**/*.JPG', recursive=True) +
              glob.glob(f'{val_dir}/**/*.png', recursive=True))

    if not images:
        log('  本地无验证集图片，跳过冒烟测试（Endpoint 已 InService）')
        return

    sample = images[0]
    log(f'  测试图片: {sample}')
    with open(sample, 'rb') as f:
        body = json.dumps({'image': base64.b64encode(f.read()).decode()})

    rt = boto3.client('sagemaker-runtime', region_name=REGION)
    resp = rt.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType='application/json',
        Body=body,
    )
    result = json.loads(resp['Body'].read())
    top_class = result['names'][result['top1']]
    confidence = result['top1conf']
    log(f'  推理结果: {top_class} ({confidence:.2%})')
    log(f'  完整结果: {result}')


# ── 主流程 ────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--yolo-only', action='store_true', help='只训 YOLO（快）')
    p.add_argument('--skip-training', metavar='MODEL_URI', default=None,
                   help='跳过训练，直接用指定的 model_uri 重新打包部署')
    args = p.parse_args()

    models_str = 'yolo' if args.yolo_only else 'yolo,resnet50,mobilenet,cnn'

    s3 = boto3.client('s3', region_name=REGION)
    sm = boto3.client('sagemaker', region_name=REGION)

    log('=' * 60)
    log('马铃薯叶片疾病 AI — 完整端到端流水线')
    log(f'训练模型: {models_str}')
    log('=' * 60)

    if args.skip_training:
        model_uri = args.skip_training
        log(f'跳过训练，使用: {model_uri}')
    else:
        source_uri = upload_source(s3)
        job_name   = submit_training_job(sm, source_uri, models_str)
        model_uri  = wait_for_training(sm, job_name)

    deploy_uri = repack_model(s3, model_uri)
    model_name = create_model(sm, deploy_uri)
    cfg_name   = create_endpoint_config(sm, model_name)
    update_endpoint(sm, cfg_name)
    smoke_test()

    log('=' * 60)
    log('✓ 全流程完成！')
    log(f'  Endpoint: {ENDPOINT_NAME}  (InService)')
    log(f'  Deploy 包: {deploy_uri}')
    log('=' * 60)


if __name__ == '__main__':
    main()
