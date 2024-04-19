import cv2

# 分類器
cascade_path = "opencv-master/data/haarcascades/haarcascade_frontalface_default.xml"

# 画像ファイル
image_input = "input/face.jpeg"

# 出力ファイル
image_output = "output/face.jpeg"

# ファイル読み込み
image = cv2.imread(image_input)

# グレースケール変換
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 特徴量読み込み
cascade = cv2.CascadeClassifier(cascade_path)

# 顔検出
face_detect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

# 検出した場合
if len(face_detect) > 0:

    # 検出した顔を囲む矩形の作成
    for x, y, w, h in face_detect:
        cv2.rectangle(image, (x, y), (x + w, y + h), (127, 255, 0), 2)

    # 認識結果の保存
    cv2.imwrite(image_output, image)