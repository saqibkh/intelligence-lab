import argparse
import os
from moviepy.editor import AudioFileClip, ImageClip, VideoFileClip, concatenate_videoclips

def create_video_with_image(audio_path, image_path, output_path):
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    image_clip = ImageClip(image_path).set_duration(duration).resize(height=720)
    video_clip = image_clip.set_audio(audio_clip)

    video_clip.write_videofile(output_path, fps=24)

def create_video_with_video(audio_path, video_path, output_path):
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration

    original_video = VideoFileClip(video_path).without_audio()
    clips = []

    # Repeat video as many times as needed to match audio duration
    while sum([clip.duration for clip in clips]) < audio_duration:
        remaining = audio_duration - sum([clip.duration for clip in clips])
        clip = original_video.subclip(0, min(original_video.duration, remaining))
        clips.append(clip)

    final_video = concatenate_videoclips(clips).set_duration(audio_duration)
    final_video = final_video.set_audio(audio_clip)

    final_video.write_videofile(output_path, fps=24)

def main():
    parser = argparse.ArgumentParser(description="Create a video from audio and either an image or a video.")
    parser.add_argument('--audio', required=True, help="Path to the input .mp3 audio file")
    parser.add_argument('--image', help="Path to the input image file (.jpg/.jpeg)")
    parser.add_argument('--video', help="Path to the input video file to be looped")
    args = parser.parse_args()

    if not args.image and not args.video:
        parser.error("You must provide either --image or --video.")
    if args.image and args.video:
        parser.error("Provide only one of --image or --video, not both.")

    output_filename = os.path.splitext(os.path.basename(args.audio))[0] + ".mp4"

    if args.image:
        create_video_with_image(args.audio, args.image, output_filename)
    else:
        create_video_with_video(args.audio, args.video, output_filename)

if __name__ == '__main__':
    main()

