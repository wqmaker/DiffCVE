# H.264 Codec

The encoder and decoder are prebuilt CPU executables for Linux x86_64 with glibc 2.17 or later.

## Setup

After downloading the repository, run from its root:

```bash
chmod +x codec/encoder codec/decoder
```

## Encode

Provide 8-bit YUV420P input, its dimensions, frame count, frame rate, and fixed QP. Create the output directory first. Run from `codec/`:

```bash
./encoder -d encoder.cfg \
  -p InputFile=/path/to/video.yuv -p OutputFile=/path/to/video.264 \
  -p SourceWidth=832 -p SourceHeight=480 -p FramesToBeEncoded=500 -p FrameRate=50 \
  -p ProfileIDC=66 -p LevelIDC=50 -p QPISlice=37 -p QPPSlice=37 \
  -p NumberReferenceFrames=5 -p SearchRange=32 -p IntraPeriod=0 -p IDRPeriod=0 \
  -p NumberBFrames=0 -p YUVFormat=1 -p SourceBitDepthLuma=8 -p SourceBitDepthChroma=8 \
  -p ReconFile=/dev/null -p TraceFile=/dev/null \
  -p LeakyBucketParamFile=/dev/null -p StatsFile=/dev/null
```

## Decode and enhance

Set up the [model environment](../README.md#installation) and install FFmpeg. See the [main README](../README.md#inference) for decoding and enhancement commands, using `--ffmpeg` to specify the FFmpeg executable.

## Copyright

The encoder and decoder are based on [JM](https://vcgit.hhi.fraunhofer.de/jvet/JM). Their copyright and license notices are included in [COPYRIGHT_ITU.txt](COPYRIGHT_ITU.txt) and [COPYRIGHT_ISO_IEC.txt](COPYRIGHT_ISO_IEC.txt).
