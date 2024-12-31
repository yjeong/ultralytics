import cv2
from ultralytics import solutions
import yt_dlp
import tempfile
import os
import sys

# yt-dlp를 사용하여 YouTube 비디오를 임시 파일로 다운로드하는 함수
def download_youtube_video_yt_dlp(youtube_url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'outtmpl': os.path.join(tempfile.gettempdir(), '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info_dict = ydl.extract_info(youtube_url, download=True)
            temp_video_path = ydl.prepare_filename(info_dict)
            return temp_video_path
        except Exception as e:
            print(f"yt-dlp로 비디오 다운로드 중 오류 발생: {e}")
            sys.exit(1)

# YouTube 비디오 URL을 여기에 입력하세요
youtube_url = "https://www.youtube.com/watch?v=88JA9f172T0&list=LL&index=8&ab_channel=FreeFootage-Videosforcontentcreators"

# yt-dlp를 사용하여 YouTube 비디오 다운로드
video_path = download_youtube_video_yt_dlp(youtube_url)

# 다운로드한 비디오 파일로 VideoCapture 초기화
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("비디오 파일을 여는 중 오류 발생")
    sys.exit(1)

# 비디오 속성 가져오기
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, 
                                      cv2.CAP_PROP_FRAME_HEIGHT, 
                                      cv2.CAP_PROP_FPS))

# 영역 포인트 정의
region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # 사각형 영역 카운팅용

# 비디오 라이터 초기화
video_writer = cv2.VideoWriter("output/object_counting_output_n.avi", 
                               cv2.VideoWriter_fourcc(*"mp4v"), 
                               fps, (w, h))

# ObjectCounter 초기화
counter = solutions.ObjectCounter(
    show=True,  # 출력 표시
    region=region_points,  # 영역 포인트 전달
    model="yolo11n.pt",  # YOLO11 OBB 모델 사용 시 "yolo11n-obb.pt"
    # classes=[0, 2],  # 특정 클래스(예: 사람, 자동차) 카운팅 시 주석 해제 및 지정
    # show_in=True,  # 내부 카운트 표시
    # show_out=True,  # 외부 카운트 표시
    # line_width=2,  # 바운딩 박스 및 텍스트 표시 시 선 너비 조절
)

# 비디오 프레임 처리
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("비디오 프레임이 비어 있거나 비디오 처리가 성공적으로 완료되었습니다.")
        break
    # 객체 카운팅 수행
    processed_frame = counter.count(frame)
    # 처리된 프레임을 출력 비디오에 작성
    video_writer.write(processed_frame)

# 자원 해제
cap.release()
video_writer.release()
cv2.destroyAllWindows()

# 처리 후 임시 비디오 파일 삭제 (선택 사항)
try:
    os.remove(video_path)
except Exception as e:
    print(f"임시 비디오 파일 삭제 중 오류 발생: {e}")
