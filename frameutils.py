#
# utilities for manipulating, finding, and loading frames
#
import torch

def timestampToSeconds (tsStr, framerate):
  tsList = tsStr.split(":")
  if len(tsList) != 4:
    raise ValueError("Not a timestamp string: " + ptsStr)
  return int(tsList[0]) * 3600 + int(tsList[1]) * 60 + int(tsList[2]) + float(tsList[3]) / framerate

def randomFrameTime (fps, limit):
  """
  Selects a random frame, returning its timestamp in seconds.

  * fps - number of frames per second (used to ensure the timestamp returned exists)
  * limit - the timestamp of the frame after the last frame to be considered

  """
  limitInt = round(limit * fps)
  frameInt = random.randrange(0,limitInt)
  frameTime = frameInt / float(fps)
  #print (f"limitInt={limitInt} frameInt={frameInt} frameTime={frameTime}")
  return frameTime

def loadFrames (video, transitionTime, fps, startFramesBeforeTransition = 0, framesToMerge=[1]):
  """
  Load a group of frames from video around transitionTime seconds (converted to frames via the fps param).
  These parameters are not self-expanatory:
  * startFramesBeforeTransition (integer) to identify the number of frames to start prior to transitionTime
  * framesToMerge (a list of integers) to identify the number of frames to merge into each output frame

  Returns a tensor of unnormalized image data containing one frame for each entry in the framesToMerge list.
  """
  targetStart = transitionTime - (startFramesBeforeTransition + 1.0)/fps # start a frame early because we discard the target frame when we find it
  # print (f"{transition[0]}: {transitionTime}, starting at {targetStart}")

  video.seek(targetStart)
  # search for the target frame
  testFrame = next(video)
  while testFrame['pts'] + 0.001 < targetStart:  # add a fudge factor to pts to avoid rounding errors causing issues
    testFrame = next(video)
  # having found the target frame we now ignore it (hence the 1 added above)
  # and carry on frame by frame creating our input vector
  inputs = []
  for col in range(0,len(framesToMerge)):
    if framesToMerge[col] == 1:
      frame = next(video)['data']
    else:
      frames = [next(video)['data'] for i in range(framesToMerge[col])]
      frame = torch.mean(torch.stack(frames), 0, dtype=torch.float16).byte()
    if torch.cuda.is_available():
      frame = frame.to("cuda")

    inputs.append(frame)

  #print(f"Row{counter} col{col}: pts={testFrame['pts']}")
  return torch.stack (inputs)
