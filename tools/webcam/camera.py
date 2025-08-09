import av
import cv2 as cv

class Camera:
  def __init__(self, cam_type_state, stream_type, camera_id):
    try:
      camera_id = int(camera_id)
    except ValueError: # allow strings, ex: /dev/video0
      pass
    self.cam_type_state = cam_type_state
    self.stream_type = stream_type
    self.cur_frame_id = 0

    print(f"Opening {cam_type_state} at {camera_id}")

    self.cap = cv.VideoCapture(camera_id)

    self.cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))

    self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280.0)
    self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720.0)
    self.cap.set(cv.CAP_PROP_FPS, 25.0)

    self.W = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
    self.H = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    self.fps = self.cap.get(cv.CAP_PROP_FPS)

    self.cap.read()

    print(f"W: {self.W} H: {self.H} fps: {self.fps} backend: {self.cap.getBackendName()}")

  @classmethod
  def bgr2nv12(self, bgr):
    frame = av.VideoFrame.from_ndarray(bgr, format='bgr24')
    return frame.reformat(format='nv12').to_ndarray()

  def read_frames(self):
    try:
      while True:
        sts , frame = self.cap.read()
        if not sts:
          print ("cv no sts")
          break
        # Rotate the frame 180 degrees (flip both axes)
        frame = cv.flip(frame, -1)
        yuv = Camera.bgr2nv12(frame)
        yield yuv.data.tobytes()
    except cv.error as error:
      print(f"cv error: {error}")
    self.cap.release()
