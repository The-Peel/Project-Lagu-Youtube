import os
import numpy as np
import matplotlib
# Use Agg backend for non-interactive video generation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy import *

# ================= CONFIGURATION =================
REPLAY_COUNT = 1
TRANSITION_DURATION = 1.0

# Explicit Audio Playlist
AUDIO_FILES = [
    "1.wav"
]

# File definitions
OPENING_VIDEO = "open.mp4"
STATIC_IMAGE = "static.jpeg"
OUTPUT_FILE = "final_output.mp4"

# VIDEO SETTINGS
VIDEO_SIZE = (1920, 1080) # 1080p
FPS = 24

# WAVEFORM VISUALIZATION SETTINGS
WAVE_COLOR = '#1e3a8a' # Dark blue
WAVE_LINE_WIDTH = 3
# Figure pixel width = 1920 / 100 dpi = 19.2 inches
# Reduced height for smaller waveform at bottom
FIG_SIZE = (25,2) 
WAVE_AMPLITUDE_GAIN = 3.5 # Multiply audio values to make wave bigger
WAVE_Y_LIMIT = 2.5        # Y-axis limit (-1 to 1 is standard audio, but we might want to zoom)

# ================= HELPER =================
def mplfig_to_rgba_image(fig):
    """
    Converts a matplotlib figure to a RGBA frame (numpy array).
    Preserves alpha channel for transparency.
    """
    fig.canvas.draw()
    # Get RGBA buffer
    try:
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    except AttributeError:
        # Fallback for older matplot/backends if needed, though Agg supports buffer_rgba
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
        # Note: argb needs reordering potentially, but buffer_rgba is standard now.
    
    w, h = fig.canvas.get_width_height()
    data = data.reshape((h, w, 4))
    return data

# ================= MAIN LOGIC =================

def make_music_video():
    print("=== STARTING VIDEO GENERATION ===")
    
    # 1. AUDIO SETUP
    # -----------------------------
    missing_files = [f for f in AUDIO_FILES if not os.path.exists(f)]
    if missing_files:
        print(f"ERROR: Missing files: {missing_files}")
        return

    print(f"Building audio playlist ({REPLAY_COUNT} replays)...")
    audio_clips = [AudioFileClip(f) for f in AUDIO_FILES]
    final_audio = concatenate_audioclips(audio_clips * REPLAY_COUNT)
    total_audio_duration = final_audio.duration
    print(f"Total Audio Duration: {total_audio_duration:.2f}s")


    # 2. OPENING VIDEO
    # -----------------------------
    if not os.path.exists(OPENING_VIDEO):
        print(f"ERROR: {OPENING_VIDEO} not found.")
        return
        
    print(f"Loading and resizing {OPENING_VIDEO}...")
    # Resize to 1920x1080
    intro_clip = (VideoFileClip(OPENING_VIDEO)
                  .without_audio()
                  .resized(VIDEO_SIZE)) 
    
    intro_duration = intro_clip.duration


    # 3. STATIC BACKGROUND
    # -----------------------------
    # Calculate duration
    main_duration = total_audio_duration - intro_duration + TRANSITION_DURATION
    if main_duration < 0: main_duration = 0.1 # Edge case safety
    
    print(f"Loading and resizing {STATIC_IMAGE}...")
    bg_clip = (ImageClip(STATIC_IMAGE)
               .resized(VIDEO_SIZE)
               .with_duration(main_duration))


    # 4. WAVEFORM OVERLAY
    # -----------------------------
    print("Initializing waveform generator...")
    
    # Create transparent figure
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='none')
    fig.patch.set_alpha(0.0) # Ensure figure background is transparent
    
    def make_waveform_frame(t):
        # Calculate real audio time
        video_start_offset = max(0, intro_duration - TRANSITION_DURATION)
        audio_time = video_start_offset + t
        
        # Setup Plot Area
        ax.clear()
        ax.set_axis_off()
        # Set Y limits
        ax.set_ylim(-WAVE_Y_LIMIT, WAVE_Y_LIMIT) 
        # Don't set X limits - let matplotlib auto-scale for proper waveform display
        # Transparent axes
        ax.patch.set_alpha(0.0)

        # Get Audio Chunk - larger window for visible waveform
        window = 0.2 # 200ms window for more samples
        start_t = min(audio_time, total_audio_duration)
        end_t = min(audio_time + window, total_audio_duration)
        
        chunk = np.zeros(1000) # Default silence
        if end_t - start_t > 0.001:
            try:
                # Get snippet
                snippet = final_audio.subclipped(start_t, end_t).to_soundarray()
                if len(snippet) > 0:
                    # Convert to mono first
                    if snippet.ndim > 1:
                        snippet = snippet.mean(axis=1)
                    chunk = snippet
            except:
                pass
        
        # Apply gain
        chunk = chunk * WAVE_AMPLITUDE_GAIN
        
        # Downsample if too many points (for performance)
        if len(chunk) > 2000:
            chunk = chunk[::len(chunk)//2000]
        
        # Plot simple line with transparency
        x = np.arange(len(chunk))
        ax.plot(x, chunk, color=WAVE_COLOR, linewidth=2, alpha=0.5)
        
        # Return RGBA
        return mplfig_to_rgba_image(fig)

    print("Creating waveform video clip...")
    # Create clip from function
    waveform_clip = VideoClip(make_waveform_frame, duration=main_duration)
    
    # IMPORTANT: Tell MoviePy this clip has an alpha channel (mask)
    # When we create a VideoClip from a MakeFrame that returns 4 channels (RGBA),
    # MoviePy automatically handles it, but let's be implicit if needed.
    # Actually, VideoClip automatically detects 4 channels.
    
    # Position: Bottom Center
    waveform_clip = waveform_clip.with_position(('center', 'bottom'))
    
    # Composite: Static BG + Waveform
    print("Compositing Main Clip...")
    main_clip = CompositeVideoClip([bg_clip, waveform_clip])
    
    # Add fade in transition to main clip
    main_clip = main_clip.with_effects([vfx.CrossFadeIn(TRANSITION_DURATION)])


    # 5. FINAL ASSEMBLY
    # -----------------------------
    print("Concatenating clips...")
    final_video = concatenate_videoclips(
        [intro_clip, main_clip], 
        method="compose", # Required for transparency/crossfade
        padding=-TRANSITION_DURATION
    )
    
    # Set Audio
    final_video = final_video.with_audio(final_audio)
    final_video = final_video.with_duration(total_audio_duration)

    # Render
    print(f"Rendering to {OUTPUT_FILE}...")
    final_video.write_videofile(
        OUTPUT_FILE, 
        fps=FPS, 
        codec='libx264', 
        audio_codec='aac',
        threads=16 # Optional: speed up
    )
    
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    make_music_video()
