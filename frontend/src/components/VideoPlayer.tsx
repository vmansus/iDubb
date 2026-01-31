import { useRef, useState, useEffect, useImperativeHandle, forwardRef, memo } from 'react'
import { Play, Pause, Volume1, VolumeX, Maximize, Download, Loader2 } from 'lucide-react'

// Preview video component for progress bar hover
// Supports both horizontal (16:9) and vertical (9:16) videos
const PreviewVideo = memo(function PreviewVideo({ src, seekTime }: { src: string; seekTime: number }) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [isLoaded, setIsLoaded] = useState(false)
  const [isVertical, setIsVertical] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video || !isLoaded) return

    // Only seek if the difference is significant
    if (Math.abs(video.currentTime - seekTime) > 0.3) {
      video.currentTime = seekTime
    }
  }, [seekTime, isLoaded])

  const handleLoadedData = () => {
    const video = videoRef.current
    if (video) {
      // Detect if video is vertical based on natural dimensions
      const aspectRatio = video.videoWidth / video.videoHeight
      setIsVertical(aspectRatio < 0.9)
    }
    setIsLoaded(true)
  }

  // For vertical videos: 90px wide x 160px tall
  // For horizontal videos: 160px wide x 90px tall
  const previewWidth = isVertical ? 90 : 160
  const previewHeight = isVertical ? 160 : 90

  return (
    <div
      className="bg-gray-900 relative"
      style={{ width: previewWidth, height: previewHeight }}
    >
      <video
        ref={videoRef}
        src={src}
        className="w-full h-full object-contain"
        muted
        preload="auto"
        onLoadedData={handleLoadedData}
      />
      {!isLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-900">
          <Loader2 className="w-5 h-5 text-white/50 animate-spin" />
        </div>
      )}
    </div>
  )
})

export interface VideoPlayerHandle {
  seekTo: (time: number) => void
  play: () => void
  pause: () => void
  getCurrentTime: () => number
}

interface VideoPlayerProps {
  src: string
  maxHeight?: string
  showDownload?: boolean
  onTimeUpdate?: (time: number) => void
  className?: string
}

const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(function VideoPlayer(
  { src, maxHeight = '500px', showDownload = true, onTimeUpdate, className = '' },
  ref
) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const videoContainerRef = useRef<HTMLDivElement>(null)
  const progressBarRef = useRef<HTMLDivElement>(null)

  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)
  const [showControls, setShowControls] = useState(true)
  const [showPreview, setShowPreview] = useState(false)
  const [previewTime, setPreviewTime] = useState(0)
  const [previewPosition, setPreviewPosition] = useState(0)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const [isVertical, setIsVertical] = useState(false)

  // Expose methods to parent
  useImperativeHandle(ref, () => ({
    seekTo: (time: number) => {
      if (videoRef.current) {
        videoRef.current.currentTime = time
        videoRef.current.play()
      }
    },
    play: () => videoRef.current?.play(),
    pause: () => videoRef.current?.pause(),
    getCurrentTime: () => videoRef.current?.currentTime || 0,
  }))

  // Listen for fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener('fullscreenchange', handleFullscreenChange)
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange)
  }, [])

  // Format time as MM:SS
  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}:${String(s).padStart(2, '0')}`
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      const time = videoRef.current.currentTime
      setCurrentTime(time)
      onTimeUpdate?.(time)
    }
  }

  return (
    <div
      ref={videoContainerRef}
      className={`relative bg-black rounded-lg group ${
        isFullscreen ? 'flex items-center justify-center' : 'overflow-hidden'
      } ${isVertical && !isFullscreen ? 'flex justify-center mx-auto' : ''} ${className}`}
      style={{
        // For vertical videos, limit the container width to maintain proper aspect ratio
        maxWidth: isVertical && !isFullscreen ? `calc(${maxHeight} * 9 / 16)` : undefined,
      }}
      onMouseEnter={() => setShowControls(true)}
      onMouseLeave={() => !isPlaying && setShowControls(true)}
    >
      <video
        ref={videoRef}
        src={src}
        className={`object-contain ${isFullscreen ? 'max-h-full w-auto' : ''} ${
          isVertical && !isFullscreen ? 'h-full w-auto' : 'w-full'
        }`}
        style={{ maxHeight: isFullscreen ? undefined : maxHeight, height: isVertical && !isFullscreen ? maxHeight : undefined }}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={() => {
          if (videoRef.current) {
            setDuration(videoRef.current.duration)
            // Detect vertical video based on natural dimensions
            const aspectRatio = videoRef.current.videoWidth / videoRef.current.videoHeight
            setIsVertical(aspectRatio < 0.9)
          }
        }}
        onPlay={() => setIsPlaying(true)}
        onPause={() => setIsPlaying(false)}
        onEnded={() => setIsPlaying(false)}
        onClick={() => {
          if (videoRef.current) {
            if (isPlaying) {
              videoRef.current.pause()
            } else {
              videoRef.current.play()
            }
          }
        }}
      />

      {/* Video Controls Overlay */}
      <div
        className={`absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4 transition-opacity duration-300 ${
          showControls ? 'opacity-100' : 'opacity-0'
        }`}
      >
        {/* Progress Bar with Preview */}
        <div
          ref={progressBarRef}
          className="mb-3 relative py-3 -my-3 cursor-pointer"
          onMouseMove={(e) => {
            if (!progressBarRef.current || !duration) return
            const rect = progressBarRef.current.getBoundingClientRect()
            const x = e.clientX - rect.left
            const percentage = Math.max(0, Math.min(1, x / rect.width))
            const time = percentage * duration
            setPreviewTime(time)
            setPreviewPosition(x)
            setShowPreview(true)
          }}
          onMouseLeave={() => setShowPreview(false)}
        >
          {/* Preview Tooltip */}
          <div
            className={`absolute bottom-full mb-4 pointer-events-none transition-opacity duration-100 ${
              showPreview && duration > 0 ? 'opacity-100' : 'opacity-0 invisible'
            }`}
            style={{
              left: `${Math.max(85, Math.min(previewPosition, (progressBarRef.current?.clientWidth || 0) - 85))}px`,
              transform: 'translateX(-50%)',
              zIndex: 100,
            }}
          >
            <div className="bg-black rounded-lg overflow-hidden shadow-2xl border border-white/40">
              <PreviewVideo src={src} seekTime={previewTime} />
              <div className="text-white text-xs text-center py-1.5 bg-black font-mono">
                {formatTime(previewTime)}
              </div>
            </div>
            {/* Arrow pointing down */}
            <div
              className="absolute left-1/2 -translate-x-1/2 w-0 h-0"
              style={{
                bottom: '-6px',
                borderLeft: '6px solid transparent',
                borderRight: '6px solid transparent',
                borderTop: '6px solid black',
              }}
            />
          </div>
          <input
            type="range"
            min={0}
            max={duration || 100}
            value={currentTime}
            onChange={(e) => {
              const time = parseFloat(e.target.value)
              if (videoRef.current) {
                videoRef.current.currentTime = time
                setCurrentTime(time)
              }
            }}
            className="w-full h-1.5 bg-white/30 rounded-lg appearance-none cursor-pointer hover:h-2 transition-all
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-white
              [&::-webkit-slider-thumb]:cursor-pointer
              [&::-webkit-slider-thumb]:shadow-md
              [&::-webkit-slider-thumb]:hover:w-4
              [&::-webkit-slider-thumb]:hover:h-4
              [&::-webkit-slider-thumb]:transition-all"
          />
        </div>

        {/* Controls */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            {/* Play/Pause */}
            <button
              onClick={() => {
                if (videoRef.current) {
                  if (isPlaying) {
                    videoRef.current.pause()
                  } else {
                    videoRef.current.play()
                  }
                }
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
            >
              {isPlaying ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
            </button>

            {/* Volume */}
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  if (videoRef.current) {
                    videoRef.current.muted = !isMuted
                    setIsMuted(!isMuted)
                  }
                }}
                className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              >
                {isMuted || volume === 0 ? (
                  <VolumeX className="h-4 w-4" />
                ) : (
                  <Volume1 className="h-4 w-4" />
                )}
              </button>
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={isMuted ? 0 : volume}
                onChange={(e) => {
                  const vol = parseFloat(e.target.value)
                  if (videoRef.current) {
                    videoRef.current.volume = vol
                    setVolume(vol)
                    if (vol > 0) setIsMuted(false)
                  }
                }}
                className="w-20 h-1 bg-white/30 rounded-lg appearance-none cursor-pointer
                  [&::-webkit-slider-thumb]:appearance-none
                  [&::-webkit-slider-thumb]:w-2
                  [&::-webkit-slider-thumb]:h-2
                  [&::-webkit-slider-thumb]:rounded-full
                  [&::-webkit-slider-thumb]:bg-white
                  [&::-webkit-slider-thumb]:cursor-pointer"
              />
            </div>

            {/* Time Display */}
            <span className="text-white text-sm">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>
          </div>

          <div className="flex items-center space-x-2">
            {/* Fullscreen */}
            <button
              onClick={() => {
                if (videoContainerRef.current) {
                  if (document.fullscreenElement) {
                    document.exitFullscreen()
                  } else {
                    videoContainerRef.current.requestFullscreen()
                  }
                }
              }}
              className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
            >
              <Maximize className="h-4 w-4" />
            </button>

            {/* Download */}
            {showDownload && (
              <a
                href={src}
                download
                className="p-2 text-white hover:bg-white/20 rounded-full transition-colors"
              >
                <Download className="h-4 w-4" />
              </a>
            )}
          </div>
        </div>
      </div>

      {/* Play Button Overlay (when paused) */}
      {!isPlaying && (
        <div
          className="absolute inset-0 flex items-center justify-center cursor-pointer"
          onClick={() => videoRef.current?.play()}
        >
          <div className="p-4 bg-white/20 rounded-full backdrop-blur-sm hover:bg-white/30 transition-colors">
            <Play className="h-12 w-12 text-white" fill="white" />
          </div>
        </div>
      )}
    </div>
  )
})

export default VideoPlayer
