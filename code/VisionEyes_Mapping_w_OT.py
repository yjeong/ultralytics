import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import yt_dlp
import tempfile
import os

def download_youtube_video(youtube_url, download_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': os.path.join(download_path, 'downloaded_video.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    # 찾은 파일 경로 반환
    for ext in ['mp4', 'm4a']:
        file_path = os.path.join(download_path, f'downloaded_video.{ext}')
        if os.path.exists(file_path):
            return file_path
    raise FileNotFoundError("다운로드된 비디오 파일을 찾을 수 없습니다.")

# 유튜브 URL을 여기에 입력하세요
youtube_url = "https://www.youtube.com/watch?v=88JA9f172T0&list=LL&index=8&ab_channel=FreeFootage-Videosforcontentcreators"

# 임시 디렉토리 생성
with tempfile.TemporaryDirectory() as tmpdirname:
    try:
        video_path = download_youtube_video(youtube_url, tmpdirname)
        print(f"비디오가 다운로드되었습니다: {video_path}")
    except Exception as e:
        print(f"비디오 다운로드 중 오류 발생: {e}")
        exit(1)

    # YOLO 모델 로드
    model = YOLO("yolo11n.pt")

    # 비디오 캡처 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오 파일을 열 수 없습니다.")
        exit(1)

    # 비디오 정보 가져오기
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30  # FPS 정보가 없을 경우 기본값 설정

    # 비디오 작성기 설정
    out = cv2.VideoWriter("visioneye-pinpoint_n.avi",
                          cv2.VideoWriter_fourcc(*"MJPG"),
                          fps, (w, h))

    center_point = (-10, h)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("비디오 프레임이 비어 있거나 비디오 처리가 완료되었습니다.")
            break

        annotator = Annotator(frame, line_width=2)

        # YOLO 모델을 사용하여 객체 추적
        results = model.track(frame, persist=True)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                annotator.box_label(box, label=str(track_id), color=colors(int(track_id)))
                annotator.visioneye(box, center_point)

        # 결과 프레임 저장 및 표시
        out.write(frame)
        cv2.imshow("visioneye-pinpoint", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 자원 해제
    out.release()
    cap.release()
    cv2.destroyAllWindows()
