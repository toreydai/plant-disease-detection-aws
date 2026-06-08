#!/usr/bin/env python3
"""从 S3 val 集随机取 100 张图，逐一调 SageMaker Endpoint，统计准确率。"""
import base64, io, json, os, random, sys, time
import boto3
from collections import defaultdict

REGION        = os.environ.get('AWS_REGION', 'us-east-1')
_account      = boto3.client('sts', region_name=REGION).get_caller_identity()['Account']
BUCKET        = os.environ.get('SAGEMAKER_BUCKET', f'sagemaker-{REGION}-{_account}')
VAL_PREFIX    = 'potato-demo/data/val/'
ENDPOINT_NAME = 'potato-disease-demo'
N             = 100

s3 = boto3.client('s3', region_name=REGION)
rt = boto3.client('sagemaker-runtime', region_name=REGION)

# ── 1. 列出所有 val 图片 ──────────────────────────────────────────────
print('列出 val 集图片...', flush=True)
paginator = s3.get_paginator('list_objects_v2')
all_images = []
for page in paginator.paginate(Bucket=BUCKET, Prefix=VAL_PREFIX):
    for obj in page.get('Contents', []):
        key = obj['Key']
        if key.lower().endswith(('.jpg', '.jpeg', '.png')):
            # key 形如 potato-demo/data/val/Early Blight/xxx.jpg
            parts = key.replace(VAL_PREFIX, '').split('/')
            if len(parts) == 2:
                all_images.append({'key': key, 'label': parts[0]})

print(f'val 集共 {len(all_images)} 张，随机抽 {N} 张', flush=True)
samples = random.sample(all_images, min(N, len(all_images)))

# ── 2. 逐张推理 ────────────────────────────────────────────────────────
results = []
errors  = 0
t0 = time.time()

for i, item in enumerate(samples, 1):
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=item['key'])
        img_bytes = obj['Body'].read()
        body = json.dumps({'image': base64.b64encode(img_bytes).decode()})

        resp = rt.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=body,
        )
        pred = json.loads(resp['Body'].read())
        predicted = pred['names'][pred['top1']]
        conf      = pred['top1conf']
        correct   = (predicted == item['label'])
        results.append({
            'label': item['label'],
            'predicted': predicted,
            'conf': conf,
            'correct': correct,
        })
        mark = '✅' if correct else '❌'
        if i % 10 == 0 or not correct:
            elapsed = time.time() - t0
            print(f'[{i:3d}/100] {mark} 真值={item["label"]:<15} 预测={predicted:<15} 置信={conf:.1%}  '
                  f'({elapsed:.0f}s)', flush=True)
    except Exception as e:
        errors += 1
        print(f'[{i:3d}/100] ⚠️  ERROR: {e}', flush=True)

# ── 3. 汇总 ───────────────────────────────────────────────────────────
total_time = time.time() - t0
correct_total = sum(r['correct'] for r in results)
n_tested = len(results)

print('\n' + '='*55)
print(f'总体准确率: {correct_total}/{n_tested} = {correct_total/n_tested:.1%}')
print(f'总耗时: {total_time:.1f}s  均值: {total_time/n_tested:.2f}s/张')
if errors:
    print(f'错误: {errors} 张')
print()

# 按类别统计
by_class = defaultdict(lambda: {'total': 0, 'correct': 0, 'wrong_as': defaultdict(int)})
for r in results:
    by_class[r['label']]['total'] += 1
    if r['correct']:
        by_class[r['label']]['correct'] += 1
    else:
        by_class[r['label']]['wrong_as'][r['predicted']] += 1

print(f'{"类别":<18} {"样本数":>6} {"正确":>6} {"准确率":>8}  错误详情')
print('-'*55)
for cls in sorted(by_class):
    d = by_class[cls]
    acc = d['correct'] / d['total']
    wrong_str = ', '.join(f'{k}×{v}' for k, v in d['wrong_as'].items()) or '-'
    print(f'{cls:<18} {d["total"]:>6} {d["correct"]:>6} {acc:>8.1%}  {wrong_str}')

# 低置信度错误样本
wrong = [r for r in results if not r['correct']]
if wrong:
    print(f'\n错误样本({len(wrong)}张):')
    for r in wrong:
        print(f'  真值={r["label"]:<15} 预测={r["predicted"]:<15} 置信={r["conf"]:.1%}')
