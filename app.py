import os
import sys
import logging
import json

import base64
import hashlib
import hmac

import urllib.request, urllib.parse

import boto3

import numpy as np
import cv2
from keras_mnist import kerasPredict

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('LINE_CHANNEL_SECRET', None)
channel_access_token = os.getenv('LINE_CHANNEL_ACCESS_TOKEN', None)
if channel_secret is None:
    logger.error('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    logger.error('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

s3 = boto3.client('s3', region_name=os.environ['S3_REGION_NAME'])
bucket = os.environ['S3_BUCKET_NAME']


def handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """
    logger.info(json.dumps(event))

    try:
        body = event.get('body', '')  # Request body string
        hash = hmac.new(channel_secret.encode('utf-8'), body.encode('utf-8'), hashlib.sha256).digest()
        signature = base64.b64encode(hash).decode('utf-8')
        # Compare X-Line-Signature request header and the signature
        headers = event.get('headers', '')
        if signature != headers.get('X-Line-Signature', '') and signature != headers.get('x-line-signature', ''):
            logger.error('Validate Error')
            return {'statusCode': 403, 'body': '{}'}
        
        for event_data in json.loads(body).get('events', []):
            message_body = [{
                'type': 'text',
                'text': '手書き数字を認識するよ。\n手書き数字の画像を送ってね。',
                'quickReply': {
                    'items': [{
                        'type': 'action',
                        'action': {
                            'type': 'message',
                            'label': 'やめる',
                            'text': 'やめる'
                        }
                    },
                    {
                        'type': 'action',
                        'action': {
                            'type': 'camera',
                            'label': 'カメラを起動する'
                        }
                    },
                    {
                        'type': 'action',
                        'action': {
                            'type': 'cameraRoll',
                            'label': 'カメラロールを開く'
                        }
                    }]
                }
            }]

            if event_data['type'] == 'message':
                if event_data['message']['type'] == 'image':
                    if event_data['message']['contentProvider']['type'] == 'line':
                        url = 'https://api-data.line.me/v2/bot/message/' + str(event_data['message']['id']) + '/content'
                        headers = {
                            'Authorization': 'Bearer ' + channel_access_token,
                        }
                        req = urllib.request.Request(url=url, headers=headers)
                        with urllib.request.urlopen(req) as res:
                            res_body = res.read()    
                            result = getAnswerFromKeras(res_body)
                            logger.info(result)

                            if result == '':
                                message_body = [{
                                    'type': 'text',
                                    'text': '数字が見つかりません。'
                                }]
                            else:
                                message_body = [{
                                    'type': 'text',
                                    'text': '数字は' + str(result) + 'だよ。'
                                }]
                    else:
                        continue
                elif event_data['message']['type'] == 'text':
                    message_text = event_data['message']['text']

                    if message_text == 'やめる':                        
                        message_body = [{
                            'type': 'text',
                            'text': 'またね。'
                        }]
                else:
                    continue
            else:
                continue
            
            url = 'https://api.line.me/v2/bot/message/reply'
            headers = {
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + channel_access_token,
            }
            body = {
                'replyToken': event_data['replyToken'],
                'messages': message_body
            }
            logger.info(message_body)
            req = urllib.request.Request(url=url, data=json.dumps(body).encode('utf-8'), method='POST', headers=headers)
            with urllib.request.urlopen(req) as res:
                res_body = res.read().decode('utf-8')
                if res_body != '{}':
                    logger.info(res_body)
        
        return {'statusCode': 200, 'body': '{}'}
    
    except Exception as err:
        logger.error(err)
        return {'statusCode': 500, 'body': '{}'}


def getAnswerFromKeras(req):
    prepared_image = preprocess(req)
    if prepared_image is None:
        return ''
    result = kerasPredict.result(prepared_image)
    return result


def preprocess(img, blank=253, min_size=300, padding=3, new_size=(28, 28)):
    nparray = np.frombuffer(img, dtype=np.uint8)
#    nparray = np.fromstring(base64.b64decode(img), np.uint8)
    img_gray = cv2.imdecode(nparray, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('/tmp/img_gray.jpg', img_gray)
    s3.upload_file(Filename='/tmp/img_gray.jpg', Bucket=bucket, Key='img_gray.jpg')

    height, width = img_gray.shape[:2]
    size = min(height, width)
    top = (height - size) // 2
    bottom = top + size
    left = (width - size) // 2
    right = left + size
    img_gray2 = img_gray[top:bottom, left:right]
    cv2.imwrite('/tmp/img_gray2.jpg', img_gray2)
    s3.upload_file(Filename='/tmp/img_gray2.jpg', Bucket=bucket, Key='img_gray2.jpg')

    if np.sum(img_gray2) / img_gray2.size >= blank:
        # 白紙の場合は削除
        return None

    # ----- 画像の輪郭を抽出する -----
#    img_blur1 = cv2.GaussianBlur(img_gray2, (11, 11), 0)
#    img_inv = cv2.threshold(img_blur1, 245, 255, cv2.THRESH_BINARY_INV)[1]
    img_inv = cv2.adaptiveThreshold(
        img_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 255, 20)
    contours, hierarchy = cv2.findContours(
        img_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.imwrite('/tmp/img_inv.jpg', img_inv)
    s3.upload_file(Filename='/tmp/img_inv.jpg', Bucket=bucket, Key='img_inv.jpg')

    # ----- 最大面積の輪郭以外は背景色で埋める -----
#    img_blur2 = cv2.GaussianBlur(img_gray2, (5, 5), 0)
#    img_inv2 = cv2.threshold(img_blur2, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    img_inv2 = cv2.adaptiveThreshold(
        img_gray2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 127, 20)
    cv2.imwrite('/tmp/img_inv2.jpg', img_inv2)
    s3.upload_file(Filename='/tmp/img_inv2.jpg', Bucket=bucket, Key='img_inv2.jpg')

    if not contours:
        # 輪郭が無い場合は削除
        return None
    
    max_area_idx = np.argmax([cv2.contourArea(c) for c in contours])    # 輪郭の中での領域面積が最大のものを取得
    contours = list(contours)
    max_area = contours.pop(max_area_idx)   # 最大面積の輪郭をcontoursから取り出して削除しておく

    if cv2.contourArea(max_area) < min_size:
        # 最大面積の輪郭がmin_size未満なら削除
        return None

#    tmp_img = fill_unnecessary_area(img_inv2, contours)
    tmp_img = img_inv2
    x, y, w, h = cv2.boundingRect(max_area)

    if x >= padding and y >= padding:
        x, y, w, h = padding_position(x, y, w, h, padding)

    # ----- モルフォロジー変換 -----
    tmp_img = morph_transformation(tmp_img)
    cv2.imwrite('/tmp/tmp_img.jpg', tmp_img)
    s3.upload_file(Filename='/tmp/tmp_img.jpg', Bucket=bucket, Key='tmp_img.jpg')
    # ----- 矩形の縦横比を保ったままリサイズする -----
    cropped = tmp_img[y:(y + h), x:(x + w)]
    new_w, new_h = get_maxrect_size(w, h, new_size[0])
    new_cropped = cv2.resize(cropped, (new_w, new_h))
    # ----- 重心を画像のセンターへ移動 -----
    dst_img = move_to_center(new_cropped, new_size)
    cv2.imwrite('/tmp/dst_img.jpg', dst_img)
    s3.upload_file(Filename='/tmp/dst_img.jpg', Bucket=bucket, Key='dst_img.jpg')
    return dst_img


def fill_unnecessary_area(img, cntrs, back_color=0):
    # img内の輪郭cntrsを背景色で埋める
    for c in cntrs:
        x, y, w, h = cv2.boundingRect(c)
        img[y:(y + h), x:(x + w)] = back_color
    return img


# 抽出した矩形のパラメータ(x, y, w, h)にpad分余白を持たせる(Qiita [機械学習のためのOpenCV入門]より)
def padding_position(x, y, w, h, pad):
    return x - pad, y - pad, w + pad * 2, h + pad * 2


def morph_transformation(img):
    kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    ret_img1 = cv2.dilate(img, kernel_1, iterations=2)  # 膨張
    kernel_2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    ret_img2 = cv2.erode(ret_img1, kernel_2)  # 収縮
    return ret_img2


# 輪郭抽出した矩形の縦横比を変えない最大の辺の長さ(横, 縦)を返す
def get_maxrect_size(w, h, side_length):
    size = round(side_length * 0.75)
    aspect_ratio = w / h
    if aspect_ratio >= 1:
        return size, round(size / aspect_ratio)
    else:
        return round(size * aspect_ratio), size


def move_to_center(img, new_size):
    m = cv2.moments(img)
    # 重心
    cx = int(m['m10'] / m['m00'])
    cy = int(m['m01'] / m['m00'])
    # 移動量の計算
    tx = new_size[1] / 2 - cx
    ty = new_size[0] / 2 - cy
    # x軸方向にtx, y軸方向にty平行移動させる
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    dst = cv2.warpAffine(img, M, new_size)
    return dst
