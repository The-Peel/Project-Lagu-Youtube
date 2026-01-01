import os
import numpy as np
import matplotlib
# Use Agg backend for non-interactive video generation
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from moviepy import *

# ================= CONFIGURATION =================
REPLAY_COUNT = 2
TRANSITION_DURATION = 1.0  # Transition between intro and main content
SEGMENT_TRANSITION_DURATION = 0.5  # Transition between image segments (crossfade)

# Audio and Image Playlist (must be same length!)
# Each audio file will be paired with its corresponding image
AUDIO_FILES = [
    "synth.wav","synth1.wav"
]

IMAGE_FILES = [
    "1.jpeg","2.jpeg"
]

# File definitions
OPENING_VIDEO = "open.mp4"
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
    
    # 0. VALIDATION
    # -----------------------------
    if len(AUDIO_FILES) != len(IMAGE_FILES):
        print(f"ERROR: AUDIO_FILES ({len(AUDIO_FILES)}) and IMAGE_FILES ({len(IMAGE_FILES)}) must have the same length!")
        return
    
    missing_audio = [f for f in AUDIO_FILES if not os.path.exists(f)]
    missing_images = [f for f in IMAGE_FILES if not os.path.exists(f)]
    
    if missing_audio:
        print(f"ERROR: Missing audio files: {missing_audio}")
        return
    if missing_images:
        print(f"ERROR: Missing image files: {missing_images}")
        return
    
    # 1. AUDIO SETUP
    # -----------------------------
    print(f"Building audio playlist ({REPLAY_COUNT} replays)...")
    audio_clips = [AudioFileClip(f) for f in AUDIO_FILES]
    
    # Build playlist with replay
    playlist_audio = audio_clips * REPLAY_COUNT
    playlist_images = IMAGE_FILES * REPLAY_COUNT
    
    # Get duration of each audio segment
    audio_durations = [clip.duration for clip in playlist_audio]
    
    # Concatenate all audio
    final_audio = concatenate_audioclips(playlist_audio)
    total_audio_duration = final_audio.duration
    print(f"Total Audio Duration: {total_audio_duration:.2f}s")
    print(f"Playlist has {len(playlist_audio)} segments")

    # 2. OPENING VIDEO
    # -----------------------------
    if not os.path.exists(OPENING_VIDEO):
        print(f"ERROR: {OPENING_VIDEO} not found.")
        return
        
    print(f"Loading and resizing {OPENING_VIDEO}...")
    intro_clip = (VideoFileClip(OPENING_VIDEO)
                  .without_audio()
                  .resized(VIDEO_SIZE)) 
    
    intro_duration = intro_clip.duration

    # 3. CREATE VIDEO SEGMENTS WITH DIFFERENT IMAGES
    # -----------------------------
    print("Creating video segments with different images...")
    
    # Create transparent figure for waveform (reused for all segments)
    fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor='none')
    fig.patch.set_alpha(0.0)
    
    video_segments = []
    
    # IMPORTANT: Calculate audio offset
    # The first image segment appears at video time (intro_duration - TRANSITION_DURATION)
    # At that point, the audio has already been playing for that amount of time
    # So we need to adjust segment timings accordingly
    audio_offset = intro_duration - TRANSITION_DURATION
    
    # Adjust the first segment duration
    # If audio_offset is 7s and first audio is 10.55s, the first IMAGE should only show
    # for the remaining 3.55s to stay in sync
    adjusted_durations = audio_durations.copy()
    if len(adjusted_durations) > 0 and audio_offset > 0:
        # First segment visual duration = remaining audio after offset
        adjusted_durations[0] = max(0.1, audio_durations[0] - audio_offset)
        print(f"First segment adjusted: {audio_durations[0]:.2f}s -> {adjusted_durations[0]:.2f}s (offset: {audio_offset:.2f}s)")
    
    # Extend the LAST segment to fill the entire remaining duration
    # This prevents black screen at the end due to crossfade calculation errors
    if len(adjusted_durations) > 1:
        # Calculate how much visual time we have for all segments except the last
        total_before_last = sum(adjusted_durations[:-1])
        # Account for crossfades between segments (n-1 transitions)
        num_transitions = len(adjusted_durations) - 1
        total_before_last -= (num_transitions - 1) * SEGMENT_TRANSITION_DURATION
        
        # Remaining time for last segment
        remaining_time = total_audio_duration - total_before_last
        adjusted_durations[-1] = max(adjusted_durations[-1], remaining_time)
        print(f"Last segment extended to: {adjusted_durations[-1]:.2f}s to fill remaining duration")
    
    current_audio_time = 0  # Track where we are in the audio playlist (not absolute video time)
    
    for idx, (image_path, segment_duration, original_duration) in enumerate(zip(playlist_images, adjusted_durations, audio_durations)):
        print(f"  Segment {idx+1}/{len(playlist_images)}: {image_path} ({segment_duration:.2f}s)")
        
        # Load and resize image
        bg_clip = (ImageClip(image_path)
                   .resized(VIDEO_SIZE)
                   .with_duration(segment_duration))
        
        # Create waveform for this segment
        # Calculate when this segment appears in the video timeline
        if idx == 0:
            segment_video_start = intro_duration - TRANSITION_DURATION
        else:
            # Sum up all previous segment durations minus crossfades
            segment_video_start = intro_duration - TRANSITION_DURATION
            for i in range(idx):
                segment_video_start += adjusted_durations[i]
                if i > 0:  # Account for crossfades between segments
                    segment_video_start -= SEGMENT_TRANSITION_DURATION
        
        def make_waveform_frame(t, video_start=segment_video_start):
            # t is time within this segment (0 to segment_duration)
            # Calculate the absolute video time
            absolute_video_time = video_start + t
            
            # Since audio plays from t=0, the audio time equals video time
            audio_time = absolute_video_time
            
            # Setup Plot Area
            ax.clear()
            ax.set_axis_off()
            ax.set_ylim(-WAVE_Y_LIMIT, WAVE_Y_LIMIT) 
            ax.patch.set_alpha(0.0)

            # Get Audio Chunk
            window = 0.2
            start_t = min(audio_time, total_audio_duration)
            end_t = min(audio_time + window, total_audio_duration)
            
            chunk = np.zeros(1000)
            if end_t - start_t > 0.001:
                try:
                    snippet = final_audio.subclipped(start_t, end_t).to_soundarray()
                    if len(snippet) > 0:
                        if snippet.ndim > 1:
                            snippet = snippet.mean(axis=1)
                        chunk = snippet
                except:
                    pass
            
            # Apply gain
            chunk = chunk * WAVE_AMPLITUDE_GAIN
            
            # Downsample if needed
            if len(chunk) > 2000:
                chunk = chunk[::len(chunk)//2000]
            
            # Plot
            x = np.arange(len(chunk))
            ax.plot(x, chunk, color=WAVE_COLOR, linewidth=2, alpha=0.5)
            
            return mplfig_to_rgba_image(fig)
        
        # Create waveform clip for this segment
        waveform_clip = VideoClip(make_waveform_frame, duration=segment_duration)
        waveform_clip = waveform_clip.with_position(('center', 'bottom'))
        
        # Composite background + waveform
        segment_clip = CompositeVideoClip([bg_clip, waveform_clip])
        video_segments.append(segment_clip)
        
        # Update audio time tracker
        current_audio_time += segment_duration
    
    # 4. CONCATENATE ALL SEGMENTS WITH CROSSFADES
    # -----------------------------
    print("Concatenating all segments with crossfade transitions...")
    
    if len(video_segments) == 1:
        # Only one segment, no transitions needed
        main_video = video_segments[0]
    else:
        # Add crossfade transitions between segments
        # Apply crossfade-out to all segments except the last
        for i in range(len(video_segments) - 1):
            video_segments[i] = video_segments[i].with_effects([vfx.CrossFadeOut(SEGMENT_TRANSITION_DURATION)])
        
        # Apply crossfade-in to all segments except the first
        for i in range(1, len(video_segments)):
            video_segments[i] = video_segments[i].with_effects([vfx.CrossFadeIn(SEGMENT_TRANSITION_DURATION)])
        
        # Concatenate with overlaps
        main_video = concatenate_videoclips(
            video_segments, 
            method="compose",
            padding=-SEGMENT_TRANSITION_DURATION  # Negative padding creates overlap
        )
    
    # Apply crossfade to the beginning of main_video (for intro transition)
    main_video = main_video.with_effects([vfx.CrossFadeIn(TRANSITION_DURATION)])

    # 5. FINAL ASSEMBLY WITH INTRO
    # -----------------------------
    print("Adding intro with crossfade...")
    final_video = concatenate_videoclips(
        [intro_clip, main_video], 
        method="compose",
        padding=-TRANSITION_DURATION
    )
    
    # Calculate expected video duration accounting for transitions
    # Intro + Main, with overlaps from transitions
    num_segment_transitions = max(0, len(video_segments) - 1)
    expected_video_duration = (intro_duration + 
                               sum(audio_durations) -  # Already includes REPLAY_COUNT!
                               TRANSITION_DURATION - 
                               num_segment_transitions * SEGMENT_TRANSITION_DURATION)
    
    print(f"Expected video duration: {expected_video_duration:.2f}s")
    print(f"Audio duration: {total_audio_duration:.2f}s")
    print(f"Waveform offset: {audio_offset:.2f}s")
    
    # Audio plays from the beginning (t=0) as originally requested
    # The audio_offset is ONLY used for the waveform visualization,
    # so the waveform matches what's audible when each image appears
    final_video = final_video.with_audio(final_audio)
    
    # The total video duration should match intro + main content
    # Use the expected_video_duration which accounts for all transitions
    final_video = final_video.with_duration(expected_video_duration)
    
    print(f"Final video duration: {expected_video_duration:.2f}s")

    # 6. RENDER
    # -----------------------------
    print(f"Rendering to {OUTPUT_FILE}...")
    final_video.write_videofile(
        OUTPUT_FILE, 
        fps=FPS, 
        codec='libx264', 
        audio_codec='aac',
        threads=16
    )
    
    plt.close(fig)
    print("Done!")

if __name__ == "__main__":
    make_music_video()
