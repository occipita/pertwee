#
# image normalisation process
#

import torch
import torchvision

# default resolutions to use if unspecified (may be modified externally)
expectedHRes = 352
expectedVRes = 288

def addChannelNorm (normalisationOps, shape):
  normalisationOps.append (v2.ToDtype(torch.float16, scale=True))
  if shape[0] == 3:
    #print ("* Will map to grayscale")
    normalisationOps.append(v2.Grayscale())
  elif shape[0] != 1:
    print ("Unexpected number of channels: ", shape[0])
    raise ValueError("Invalid image format")

def addResizeNorm (normalisationOps, shape, toHRes, toVRes):
  vres = shape[1] # may need to update this if we scale the image!

  if shape[2] != toHRes:
    scaleFactor = toHRes / shape[2]
    vres = int(vres * scaleFactor)
    #print (f"* Will scale image by {scaleFactor*100:.1f}% (vres = {vres})")
    normalisationOps.append(v2.Resize([vres,toHRes]))

  if vres > toVRes:
    print (f"Image too tall! {vres}>{toVRes}")
    raise ValueError("Invalid image format")
  elif vres < toVRes:
    lb = toVRes - vres;
    lbtop = int(lb/2)
    lbbottom = lb-lbtop
    #print (f"* will add letterboxing (top={lbtop}, bottom={lbbottom})")
    normalisationOps.append(v2.Pad([0,lbtop,0,lbbottom]))

def addHalfZoomNorm (normalisationOps, shape, toHRes, toVRes):
  halfHRes = int((shape[2] + toHRes) / 2)
  halfVRes = int((shape[1] + toVRes) / 2)
  addResizeNorm (normalisationOps, shape, halfHRes, halfVRes)

def calcFrameNormalisationOps (shape):
  normalisationOps = []
  addChannelNorm (normalisationOps, shape)
  addResizeNorm (normalisationOps, shape, expectedHRes, expectedVRes)

  return normalisationOps

#
# alternative normalisation processes (for producing additional training data)
#
def altNormOpsCentralCrop (shape):
  ops = []
  addChannelNorm (ops, shape)
  ops.append (v2.CenterCrop([expectedVRes, expectedHRes]))
  return ops
def altNormOpsRandCrop (shape):
  ops = []
  addChannelNorm (ops, shape)
  ops.append (v2.RandomCrop([expectedVRes, expectedHRes]))
  return ops
def altNormOpsRandCropHalf (shape):
  ops = []
  addChannelNorm (ops, shape)
  addHalfZoomNorm (ops, shape, expectedHRes, expectedVRes)
  ops.append (v2.RandomCrop([expectedVRes, expectedHRes]))
  return ops
def hflippedNorm (norm):
  norm.append (v2.RandomHorizontalFlip(1))
  return norm
def vflippedRand (norms):
  return [v2.RandomChoice(norms), v2.RandomVerticalFlip(1)]
def vhflippedRand (norms):
  return [v2.RandomChoice(norms), v2.RandomHorizontalFlip(1), v2.RandomVerticalFlip(1)]
def altNormOpsRotated (shape):
  ops = []
  addChannelNorm (ops, shape)
  addHalfZoomNorm (ops, shape, expectedHRes, expectedVRes)
  ops.append (v2.RandomRotation(90))
  ops.append (v2.RandomCrop([expectedVRes, expectedHRes]))
  return ops
def noisyRand (norms):
  return [v2.RandomChoice(norms), v2.GaussianNoise()]


def defaultNormalisers (shape):
  normalisers = [
      v2.Compose(calcFrameNormalisationOps(shape)),
      v2.Compose(altNormOpsCentralCrop(shape)),
      v2.Compose(altNormOpsRandCrop(shape)),
      v2.Compose(altNormOpsRandCropHalf(shape)),
      v2.Compose(hflippedNorm(calcFrameNormalisationOps(shape))),
      v2.Compose(hflippedNorm(altNormOpsCentralCrop(shape))),
      v2.Compose(hflippedNorm(altNormOpsRandCrop(shape))),
      v2.Compose(hflippedNorm(altNormOpsRandCropHalf(shape))),
      v2.Compose(altNormOpsRotated(shape))]
  normalisers.append(v2.Compose(vflippedRand(normalisers[0:3])))
  normalisers.append(v2.Compose(vhflippedRand(normalisers[0:3])))
  normalisers.append(v2.Compose(noisyRand(normalisers[0:3])))
  normalisers.append(v2.Compose(noisyRand(normalisers[4:])))
  return normalisers
